import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.chisight.dense.likelihoods as likelihoods
import b3d.chisight.dense.differentiable_renderer as rendering
import rerun as rr

def normalize(weights):
    return weights / jnp.sum(weights)
def get_likelihood(renderer, color_scale = 0.05, outlier_prob = 0.05):
    return likelihoods.ArgMap(
        likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
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
    def model(background_mesh, triangle_color, camera_poses):
        """
        - background_mesh = (background_vertices, background_faces, background_colors)
        - triangle (3, 3): [v1, v2, v3]
                (will be renormalized so v1 = [0, 0, 0] and len(v1->v2) == 1.0)
        """
        (background_vertices, background_faces, background_colors) = background_mesh

        triangle_vertices = genjax.uniform(
            -20. * jnp.ones((3, 3)), 20. * jnp.ones((3, 3))
        ) @ "triangle_vertices"

        all_vertices = jnp.concatenate([background_vertices, triangle_vertices], axis=0)
        all_faces = jnp.concatenate([background_faces, jnp.arange(3).reshape((1, 3)) + background_vertices.shape[0]], axis=0)
        all_face_colors = jnp.concatenate([background_colors, jnp.array([triangle_color])], axis=0)

        (observed_rgbs, weights, attributes) = genjax.map_combinator(
            in_axes=(0, None, None, None)
        )(generate_frame)(camera_poses, all_vertices, all_faces, all_face_colors) @ "observed_rgbs"

        metadata = {
            "weights": weights,
            "attributes": attributes,
            "triangle_vertices": triangle_vertices,
        }

        return (observed_rgbs, metadata)

    return model

def rr_log_trace(
        trace, renderer,
        prefix="trace",
        frames_images_to_visualize=[0],
        frames_cameras_to_visualize=[0]
    ):
    (observed_rgbs, metadata) = trace.get_retval()
    weights, attributes = metadata["weights"], metadata["attributes"]
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
        rr.log(f"/3D/{prefix}/{t}/camera", rr.Image(observed_rgbs[t, :, :]))

    # log background
    (bv, bf, bfc) = trace.get_args()[0]
    bv_, bf_, bvc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(bv, bf, bfc)
    rr.log(f"/3D/{prefix}/background", rr.Mesh3D(
        vertex_positions=bv_, indices=bf_, vertex_colors=bvc_
    ))

    # log foreground triangle
    tv = metadata["triangle_vertices"]
    tfc = trace.get_args()[1][None, :]
    tf = jnp.array([[0, 1, 2]])
    tv_, tf_, tvc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(tv, tf, tfc)
    rr.log(f"/3D/{prefix}/foreground", rr.Mesh3D(
        vertex_positions=tv_, indices=tf_, vertex_colors=tvc_
    ))
