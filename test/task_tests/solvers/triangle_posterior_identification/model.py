import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.differentiable_renderer as rendering
from b3d.model import uniform_pose
import rerun as rr

def normalize(weights):
    return weights / jnp.sum(weights)
def get_likelihood(renderer, color_scale = 0.05, outlier_prob = 0.05):
    return b3d.likelihoods.ArgMap(
        b3d.likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
            renderer.height, renderer.width, color_scale
        ),
        lambda weights, attributes:(
            normalize(jnp.concatenate([weights[:1] + outlier_prob, weights[1:]])),
            attributes
        )
    )

def model_factory(
        renderer, likelihood,
        renderer_hyperparams
    ):
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
    def model(background_mesh, triangle, triangle_color, camera_poses):
        """
        - background_mesh = (background_vertices, background_faces, background_colors)
        - triangle (3, 3): [v1, v2, v3]
                (will be renormalized so v1 = [0, 0, 0] and len(v1->v2) == 1.0)
        """
        (background_vertices, background_faces, background_colors) = background_mesh

        # scaling and shifting is relative to v1
        triangle_xyz = genjax.uniform(jnp.zeros(3), 5. * jnp.ones(3)) @ "triangle_xyz"
        triangle_size = genjax.uniform(0., 20.) @ "triangle_size"

        triangle = triangle - triangle[0]
        triangle_transformed = triangle * triangle_size
        triangle_transformed = triangle_transformed + triangle_xyz

        all_vertices = jnp.concatenate([background_vertices, triangle_transformed], axis=0)
        all_faces = jnp.concatenate([background_faces, jnp.arange(3).reshape((1, 3)) + background_vertices.shape[0]], axis=0)
        all_face_colors = jnp.concatenate([background_colors, jnp.array([triangle_color])], axis=0)

        (observed_rgbs, weights, attributes) = genjax.map_combinator(
            in_axes=(0, None, None, None)
        )(generate_frame)(camera_poses, all_vertices, all_faces, all_face_colors) @ "observed_rgbs"

        return (observed_rgbs, weights, attributes)

    return model

def rr_log_trace(
        trace, renderer,
        prefix="trace",
        frames_images_to_visualize=[],
        frames_cameras_to_visualize=[]
    ):
    (observed_rgbs, weights, attributes) = trace.get_retval()
    avg_obs = jax.vmap(rendering.dist_params_to_average, in_axes=(0, 0, None))(weights, attributes, jnp.zeros(3))
    assert avg_obs.shape == observed_rgbs.shape
    for t in frames_images_to_visualize:
        rr.log(f"/{prefix}/rgb/{t}/observed", rr.Image(observed_rgbs[t, :, :]))
        rr.log(f"/{prefix}/rgb/{t}/average_render", rr.Image(avg_obs[t, :, :]))

    for t in frames_cameras_to_visualize:
        rr.log(f"/3D/{prefix}/{t}/camera", rr.Pinhole(
            focal_length = renderer.fx,
            width = renderer.width,
            height = renderer.height,
            principal_point = jnp.array([renderer.cx, renderer.cy])
        ))
        camera_poses = trace.get_args()[-1]
        cam_pose = camera_poses[t]
        rr.log(f"/3D/{prefix}/{t}/camera", rr.Transform3D(
            translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()
        ))

    v, f, fc = trace["vertices"], trace["faces"], trace["face_colors"]
    v_, f_, vc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(v, f, fc)
    rr.log(f"/3D/{prefix}/mesh", rr.Mesh3D(
        vertex_positions=v_, indices=f_, vertex_colors=vc_
    ))
