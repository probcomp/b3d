import jax.numpy as jnp
import jax
import b3d
import rerun as rr
from tqdm import tqdm
import optax
import b3d.chisight.dense.differentiable_renderer as rendering
from functools import partial

rr.init("gradients")
rr.connect("127.0.0.1:8812")


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def test_diff_renderer(renderer):
    # Set up OpenGL renderer
    image_width = 100
    image_height = 100
    fx = 50.0
    fy = 50.0
    cx = 50.0
    cy = 50.0
    near = 0.001
    far = 16.0
    # renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
    renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

    def render(particle_centers, particle_widths, particle_colors):
        particle_widths = jnp.abs(particle_widths)
        # Get triangle "mesh" for the scene:
        vertices, faces, vertex_colors, triangle_index_to_particle_index = jax.vmap(
            b3d.square_center_width_color_to_vertices_faces_colors
        )(
            jnp.arange(len(particle_centers)),
            particle_centers,
            particle_widths / 2,
            particle_colors,
        )
        vertices = vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)
        vertex_rgbs = vertex_colors.reshape(-1, 3)
        triangle_index_to_particle_index = triangle_index_to_particle_index.reshape(-1)
        image = rendering.render_to_average_rgbd(
            renderer,
            vertices,
            faces,
            vertex_rgbs,
            background_attribute=jnp.array([0.0, 0.0, 0.0, 0]),
        )
        return image

    render_jit = jax.jit(render)

    gt_particle_centers = jnp.array([[0.0, 0.0, 0.2]])
    gt_particle_colors = jnp.array([[1.0, 0.0, 0.0]])
    gt_particle_widths = jnp.array([[0.05]])
    gt_image = render_jit(gt_particle_centers, gt_particle_widths, gt_particle_colors)
    rr.set_time_sequence("frame", 0)
    rr.log("image", rr.Image(gt_image[..., :3]), timeless=True)

    def loss_func_rgbd(params, gt):
        image = render(
            params["particle_centers"],
            params["particle_widths"],
            params["particle_colors"],
        )
        return jnp.mean(jnp.abs(image[..., :3] - gt[..., :3])) + jnp.mean(
            jnp.abs(image[..., 3] - gt[..., 3])
        )

    loss_func_rgbd_grad = jax.jit(jax.value_and_grad(loss_func_rgbd, argnums=(0,)))

    @partial(jax.jit, static_argnums=(0,))
    def update_params(tx, params, gt_image, state):
        loss, (gradients,) = loss_func_rgbd_grad(params, gt_image)
        updates, state = tx.update(gradients, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    label_fn = map_nested_fn(lambda k, _: k)

    test_cases = [
        (
            "just position",
            optax.multi_transform(
                {
                    "particle_centers": optax.adam(1e-3),
                    "particle_widths": optax.sgd(0.0),
                    "particle_colors": optax.sgd(0.0),
                },
                label_fn,
            ),
            {
                "particle_centers": jnp.array([[0.02, 0.02, 0.4]]),
                "particle_widths": gt_particle_widths,
                "particle_colors": gt_particle_colors,
            },
        ),
        (
            "position and color",
            optax.multi_transform(
                {
                    "particle_centers": optax.adam(1e-3),
                    "particle_widths": optax.sgd(0.0),
                    "particle_colors": optax.adam(1e-2),
                },
                label_fn,
            ),
            {
                "particle_centers": jnp.array([[0.02, 0.02, 0.4]]),
                "particle_widths": gt_particle_widths,
                "particle_colors": jnp.array([[0.0, 1.0, 0.0]]),
            },
        ),
        (
            "position color and width",
            optax.multi_transform(
                {
                    "particle_centers": optax.adam(4e-3),
                    "particle_widths": optax.adam(5e-4),
                    "particle_colors": optax.adam(2e-2),
                },
                label_fn,
            ),
            {
                "particle_centers": jnp.array([[0.05, 0.05, 0.4]]),
                "particle_widths": jnp.array([0.1]),
                "particle_colors": jnp.array([[0.0, 1.0, 0.0]]),
            },
        ),
    ]

    images = []
    for title, tx, params in test_cases:
        print(title)
        pbar = tqdm(range(400))
        state = tx.init(params)
        images = [
            render_jit(
                params["particle_centers"],
                params["particle_widths"],
                params["particle_colors"],
            )
        ]
        for t in pbar:
            params, state, loss = update_params(tx, params, gt_image, state)
            pbar.set_description(f"Loss: {loss}")
            images.append(
                render_jit(
                    params["particle_centers"],
                    params["particle_widths"],
                    params["particle_colors"],
                )
            )

        print(params)
        assert jnp.allclose(params["particle_centers"], gt_particle_centers, atol=4e-3)
        assert jnp.allclose(params["particle_widths"], gt_particle_widths, atol=1e-3)
        assert jnp.allclose(params["particle_colors"], gt_particle_colors, atol=1e-2)

        viz_images = [
            b3d.multi_panel(
                [
                    b3d.get_rgb_pil_image(gt_image),
                    b3d.get_rgb_pil_image(img),
                ],
                ["GT", "Inferred"],
                label_fontsize=15,
                title_fontsize=20,
                title=title,
            )
            for img in images[::10]
        ]
        b3d.make_video_from_pil_images(
            viz_images, b3d.get_root_path() / f"assets/test_results/{title}.mp4"
        )
