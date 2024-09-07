import os

import b3d
import genjax
import jax
import jax.numpy as jnp
import optax
import rerun as rr
from tqdm import tqdm

import demos.mesh_fitting.model as m
import demos.mesh_fitting.tessellation as t


### Get initial mesh ###
def get_initial_tesselation(video_input):
    _image_width, _image_height, fx, fy, cx, cy, near, far = jnp.array(
        video_input.camera_intrinsics_rgb
    )
    fx, fy, cx, cy, near, far = (
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        float(near),
        float(far),
    )

    rgbs_full_resolution = video_input.rgb[::4] / 255.0
    vertices_2D, faces, triangle_colors = t.generate_tessellated_2D_mesh_from_rgb_image(
        rgbs_full_resolution[0]
    )
    key = jax.random.PRNGKey(0)
    depth = genjax.uniform.sample(key, 0.2, 200.0)
    vertices_3D = jnp.hstack(
        (
            (vertices_2D - jnp.array([cx, cy])) * depth / jnp.array([fx, fy]),
            jnp.ones((vertices_2D.shape[0], 1)) * depth,
        )
    )

    initial_mesh = (vertices_3D, faces, triangle_colors)
    return initial_mesh


path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.io.VideoInput.load(path)
initial_mesh = get_initial_tesselation(video_input)

### Get renderer & scaled down RGB video ###
width, height, fx, fy, cx, cy, near, far = jnp.array(
    video_input.camera_intrinsics_depth
)
width, height = int(width), int(height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)
rgbs_full_resolution = video_input.rgb[::4] / 255.0
rgbs = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs_full_resolution,
        (video_input.xyz.shape[1], video_input.xyz.shape[2], 3),
        "linear",
    ),
    0.0,
    1.0,
)

camera_poses = [
    # b3d.Pose.from_vec(jnp.concatenate([
    #     video_input.camera_positions[t],
    #     video_input.camera_quaternions[t]
    # ]))
    b3d.Pose(
        video_input.camera_positions[::4][t], video_input.camera_quaternions[::4][t]
    )
    for t in range(rgbs.shape[0])
]

initial_mesh = (
    camera_poses[0].apply(initial_mesh[0]),
    initial_mesh[1],
    initial_mesh[2],
)

### Set up model ###
frames = [0, 16, 24, 32]

mindepth, maxdepth = -1000.0, 1000.0
color_scale = 0.05
outlier_prob = 0.05


def normalize(x):
    return x / jnp.sum(x)


likelihood = b3d.chisight.dense.likelihoods.ArgMap(
    b3d.chisight.dense.likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
        renderer.height, renderer.width, color_scale
    ),
    lambda weights, attributes: (
        normalize(jnp.concatenate([weights[:1] + outlier_prob, weights[1:]])),
        attributes,
    ),
)
hyperparams = (
    b3d.chisight.dense.differentiable_renderer.DifferentiableRendererHyperparams(
        3, 1e-5, 1e-2, -1
    )
)
model = m.model_factory(
    renderer,
    likelihood,
    hyperparams,
    mindepth,
    maxdepth,
    len(frames),
    initial_mesh[0].shape[0],
    initial_mesh[1].shape[0],
)


key = jax.random.PRNGKey(0)
trace, weight = jax.jit(model.importance)(
    key,
    genjax.choice_map(
        {
            "vertices": initial_mesh[0],
            "faces": initial_mesh[1],
            "face_colors": initial_mesh[2],
            "camera_poses": genjax.vector_choice_map(
                genjax.choice(b3d.Pose.stack_poses([camera_poses[t] for t in frames]))
            ),
            "observed_rgbs": genjax.vector_choice_map(
                genjax.choice_map({"observed_rgb": rgbs[frames, ...]})
            ),
        }
    ),
    (),
)
weight

rr.init("mesh_fitting-4")
rr.connect("127.0.0.1:8812")
m.rr_log_trace(
    trace,
    renderer,
    frames_images_to_visualize=[0, 1],
    frames_cameras_to_visualize=[0, 1],
)


@jax.jit
def importance_from_vertices_colors(vertices, colors):
    trace, weight = jax.jit(model.importance)(
        key,
        genjax.choice_map(
            {
                "vertices": vertices,
                "faces": initial_mesh[1],
                "face_colors": colors,
                "camera_poses": genjax.vector_choice_map(
                    genjax.choice(
                        b3d.Pose.stack_poses([camera_poses[t] for t in frames])
                    )
                ),
                "observed_rgbs": genjax.vector_choice_map(
                    genjax.choice_map({"observed_rgb": rgbs[frames, ...]})
                ),
            }
        ),
        (),
    )
    return trace, weight


def vertices_colors_to_score(vertices, colors):
    _trace, weight = importance_from_vertices_colors(vertices, colors)
    return weight


grad_jitted = jax.jit(jax.grad(vertices_colors_to_score, argnums=(0, 1)))
value_and_grad_jitted = jax.jit(
    jax.value_and_grad(vertices_colors_to_score, argnums=(0, 1))
)

optimizer_colors = optax.adam(learning_rate=1e-2)
optimizer_vertices = optax.adam(learning_rate=1e-1)
vertices = initial_mesh[0]
colors = initial_mesh[2]
opt_state_vertices = optimizer_vertices.init(vertices)
opt_state_colors = optimizer_colors.init(colors)
timestep = 0
for dt in tqdm(range(1000)):
    timestep = timestep + 1
    score, (grads_positions, grads_colors) = value_and_grad_jitted(vertices, colors)
    updates_colors, opt_state_colors = optimizer_colors.update(
        -grads_colors, opt_state_colors
    )
    updates_vertices, opt_state_vertices = optimizer_vertices.update(
        -grads_positions, opt_state_vertices
    )
    colors = optax.apply_updates(colors, updates_colors)
    colors = jnp.clip(colors, 0, 1)
    vertices = optax.apply_updates(vertices, updates_vertices)
    if timestep % 10 == 1:
        rr.set_time_sequence("mesh_fitting_step-8", timestep)
        rr.log("logpdf", rr.Scalar(score))
    if timestep % 30 == 1:
        trace, _ = importance_from_vertices_colors(vertices, colors)
        m.rr_log_trace(
            trace,
            renderer,
            frames_images_to_visualize=[0, 1, 2, 3],
            frames_cameras_to_visualize=[0, 1, 2, 3],
        )
        if jnp.isinf(score) or jnp.isnan(score):
            print(f"Score: {score}")
