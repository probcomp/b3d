import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.chisight.dense.likelihoods as likelihoods
import b3d.chisight.dense.differentiable_renderer as rendering
import rerun as rr


def normalize(weights):
    return weights / jnp.sum(weights)


def get_diffrend_likelihood(
    renderer, renderer_hyperparams, color_scale=0.05, outlier_prob=0.05
):
    likelihood_dist = likelihoods.ArgMap(
        likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
            renderer.height, renderer.width, color_scale
        ),
        lambda weights, attributes: (
            normalize(jnp.concatenate([weights[:1] + outlier_prob, weights[1:]])),
            attributes,
        ),
    )

    @genjax.gen
    def wrapped_likelihood(mesh: b3d.Mesh, transform_World_Camera):
        weights, attributes = (
            b3d.chisight.dense.differentiable_renderer.render_to_dist_params(
                renderer,
                mesh.vertices,
                mesh.faces,
                mesh.vertex_attributes,
                renderer_hyperparams,
                transform=transform_World_Camera.inv(),
            )
        )
        obs = likelihood_dist(weights, attributes) @ "image"
        return obs, {"diffrend_output": (weights, attributes)}

    return wrapped_likelihood


def get_simple_likelihood(renderer, color_scale=0.05, outlier_prob=0.05):
    likelihood_dist = likelihoods.ArgMap(
        likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
            renderer.height, renderer.width, color_scale
        ),
        lambda colors: (
            jnp.tile(
                jnp.array([outlier_prob, 1.0 - outlier_prob]),
                (colors.shape[0], colors.shape[1], 1),
            ),
            colors[:, :, None, ...],
        ),
    )

    @genjax.gen
    def wrapped_likelihood(mesh: b3d.Mesh, transform_World_Camera):
        rgb, _ = renderer.render_attribute(
            transform_World_Camera.inv()[None, ...],
            mesh.vertices,
            mesh.faces,
            jnp.array([[0, len(mesh.faces)]]),
            mesh.vertex_attributes,
        )
        assert rgb.shape == (renderer.height, renderer.width, 3)
        obs = likelihood_dist(rgb) @ "image"
        return obs, {"rasterized_rgb": rgb}

    return wrapped_likelihood


def model_factory(likelihood):  # renderer, renderer_hyperparams): #likelihood):
    """
    The provided likelihood should be a Generative Function with
    one latent choice at address `"image"`, which accepts `mesh` as input,
    and outputs `(image, metadata)`.
    The value `image` should be sampled at `"image"`.
    """

    @genjax.gen
    def generate_frame(transform_World_Camera, vertices, faces, face_colors):
        vertices_W = vertices

        v_W, f, vc = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(
            vertices_W, faces, face_colors
        )
        mesh_W = b3d.Mesh(v_W, f, vc)

        observed_rgb, metadata = (
            likelihood(mesh_W, transform_World_Camera) @ "observed_rgb"
        )
        return (observed_rgb, metadata)

    @genjax.gen
    def model(background_mesh, triangle_color, camera_poses):
        """
        - background_mesh = (background_vertices, background_faces, background_colors)
        - triangle (3, 3): [v1, v2, v3]
                (will be renormalized so v1 = [0, 0, 0] and len(v1->v2) == 1.0)
        """
        (background_vertices, background_faces, background_colors) = background_mesh

        triangle_vertices = (
            b3d.modeling_utils.uniform(
                -20.0 * jnp.ones((3, 3)), 20.0 * jnp.ones((3, 3))
            )
            @ "triangle_vertices"
        )

        all_vertices = jnp.concatenate([background_vertices, triangle_vertices], axis=0)
        all_faces = jnp.concatenate(
            [
                background_faces,
                jnp.arange(3).reshape((1, 3)) + background_vertices.shape[0],
            ],
            axis=0,
        )
        all_face_colors = jnp.concatenate(
            [background_colors, jnp.array([triangle_color])], axis=0
        )

        (observed_rgbs, metadata) = (
            generate_frame.vmap(in_axes=(0, None, None, None))(
                camera_poses, all_vertices, all_faces, all_face_colors
            )
            @ "observed_rgbs"
        )

        metadata = {
            "likelihood_metadata": metadata,
            "triangle_vertices": triangle_vertices,
        }

        return (observed_rgbs, metadata)

    return model


def rr_log_trace(
    trace,
    renderer,
    prefix="trace",
    frames_images_to_visualize=[0],
    frames_cameras_to_visualize=[0],
):
    (observed_rgbs, metadata) = trace.get_retval()
    likelihood_metadata = metadata["likelihood_metadata"]

    if "diffrend_output" in likelihood_metadata:
        weights, attributes = likelihood_metadata["diffrend_output"]
        avg_obs = jax.vmap(rendering.dist_params_to_average, in_axes=(0, 0, None))(
            weights, attributes, jnp.zeros(3)
        )
        assert avg_obs.shape == observed_rgbs.shape

    for t in frames_images_to_visualize:
        rr.log(f"/{prefix}/rgb/{t}/observed", rr.Image(observed_rgbs[t, :, :]))
        if "diffrend_output" in likelihood_metadata:
            rr.log(f"/{prefix}/rgb/{t}/average_render", rr.Image(avg_obs[t, :, :]))

    for t in frames_cameras_to_visualize:
        rr.log(
            f"/3D/{prefix}/{t}/camera",
            rr.Pinhole(
                focal_length=float(renderer.fx),
                width=renderer.width,
                height=renderer.height,
                principal_point=jnp.array([renderer.cx, renderer.cy]),
            ),
        )
        camera_poses = trace.get_args()[-1]
        cam_pose = camera_poses[t]
        rr.log(
            f"/3D/{prefix}/{t}/camera",
            rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()),
        )
        rr.log(f"/3D/{prefix}/{t}/camera", rr.Image(observed_rgbs[t, :, :]))

    # log background
    (bv, bf, bfc) = trace.get_args()[0]
    bv_, bf_, bvc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(bv, bf, bfc)
    rr.log(
        f"/3D/{prefix}/background",
        rr.Mesh3D(vertex_positions=bv_, triangle_indices=bf_, vertex_colors=bvc_),
    )

    # log foreground triangle
    tv = metadata["triangle_vertices"]
    tfc = trace.get_args()[1][None, :]
    tf = jnp.array([[0, 1, 2]])
    tv_, tf_, tvc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(tv, tf, tfc)
    rr.log(
        f"/3D/{prefix}/foreground",
        rr.Mesh3D(vertex_positions=tv_, triangle_indices=tf_, vertex_colors=tvc_),
    )
