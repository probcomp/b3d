import b3d
import b3d.chisight.dense.differentiable_renderer as rendering
import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from b3d.modeling_utils import uniform_pose


class UniformDiscrete(genjax.ExactDensity, genjax.JAXGenerativeFunction):
    shape: any = genjax.Pytree.static()

    def sample(self, key, low, high):
        assert isinstance(low, int) or low.dtype == jnp.int32
        assert isinstance(high, int) or high.dtype == jnp.int32
        return jax.random.randint(key, self.shape, low, high)

    def logpdf(self, value, low, high):
        assert isinstance(low, int) or low.dtype == jnp.int32
        assert isinstance(high, int) or high.dtype == jnp.int32
        assert value.shape == self.shape
        lp = -jnp.log(high - low)
        return jnp.where(
            jnp.all((value < high) & (value >= low)), value.size * lp, -jnp.inf
        )


def model_factory(
    renderer,
    likelihood,
    renderer_hyperparams,
    mindepth,
    maxdepth,
    n_frames,
    n_vertices,
    n_faces,
):
    uniform_discrete = UniformDiscrete((n_faces, 3))

    @genjax.static_gen_fn
    def generate_frame(camera_pose, vertices, faces, face_colors):
        X_WC = camera_pose
        vertices_W = vertices
        # vertices_C = X_WC.inv().apply(vertices_W)

        v, f, vc = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(
            vertices_W, faces, face_colors
        )
        weights, attributes = rendering.render_to_dist_params(
            renderer, v, f, vc, renderer_hyperparams, X_WC.inv()
        )
        observed_rgb = likelihood(weights, attributes) @ "observed_rgb"
        return (observed_rgb, weights, attributes)

    @genjax.static_gen_fn
    def model():
        vertices = (
            b3d.modeling_utils.uniform(
                jnp.ones((n_vertices, 3)) * mindepth,
                jnp.ones((n_vertices, 3)) * maxdepth,
            )
            @ "vertices"
        )
        faces = uniform_discrete(0, n_vertices) @ "faces"
        face_colors = (
            b3d.modeling_utils.uniform(jnp.zeros((n_faces, 3)), jnp.ones((n_faces, 3)))
            @ "face_colors"
        )

        camera_poses = (
            genjax.map_combinator(in_axes=(0, 0))(uniform_pose)(
                jnp.ones((n_frames, 3)) * -100.0, jnp.ones((n_frames, 3)) * 100.0
            )
            @ "camera_poses"
        )

        (observed_rgbs, weights, attributes) = (
            genjax.map_combinator(in_axes=(0, None, None, None))(generate_frame)(
                camera_poses, vertices, faces, face_colors
            )
            @ "observed_rgbs"
        )

        return (observed_rgbs, weights, attributes)

    return model


def rr_log_trace(
    trace,
    renderer,
    prefix="trace",
    frames_images_to_visualize=[],
    frames_cameras_to_visualize=[],
):
    (observed_rgbs, weights, attributes) = trace.get_retval()
    avg_obs = jax.vmap(rendering.dist_params_to_average, in_axes=(0, 0, None))(
        weights, attributes, jnp.zeros(3)
    )
    assert avg_obs.shape == observed_rgbs.shape
    for t in frames_images_to_visualize:
        rr.log(f"/{prefix}/rgb/{t}/observed", rr.Image(observed_rgbs[t, :, :]))
        rr.log(f"/{prefix}/rgb/{t}/average_render", rr.Image(avg_obs[t, :, :]))

    for t in frames_cameras_to_visualize:
        rr.log(
            f"/3D/{prefix}/{t}/camera",
            rr.Pinhole(
                focal_length=renderer.fx,
                width=renderer.width,
                height=renderer.height,
                principal_point=jnp.array([renderer.cx, renderer.cy]),
            ),
        )
        cam_pose = trace["camera_poses"].inner.value[t]
        rr.log(
            f"/3D/{prefix}/{t}/camera",
            rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()),
        )

    v, f, fc = trace["vertices"], trace["faces"], trace["face_colors"]
    v_, f_, vc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(v, f, fc)
    rr.log(
        f"/3D/{prefix}/mesh",
        rr.Mesh3D(vertex_positions=v_, triangle_indices=f_, vertex_colors=vc_),
    )


def get_rgb_only_model(renderer, initial_mesh):
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
    model = model_factory(
        renderer,
        likelihood,
        hyperparams,
        mindepth,
        maxdepth,
        1,
        initial_mesh[0].shape[0],
        initial_mesh[1].shape[0],
    )
    return model
