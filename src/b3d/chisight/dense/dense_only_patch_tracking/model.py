import genjax
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr

import b3d.chisight.dense.differentiable_renderer as rendering
import b3d.utils as utils
from b3d import Pose
from b3d.modeling_utils import uniform_pose


def uniformpose_meshes_to_image_model__factory(likelihood):
    """
    This factory returns a generative function which
    (1) samples a camera pose uniformly from a spatial region,
    (2) samples a collection of object poses uniformly from a spatial region,
    and
    (3) generates an image of the collection of meshes with the given poses, from the given camera pose.

    The factory function accepts the same argument signature as `meshes_to_image_model__factory`.
    """
    meshes_to_image_model = meshes_to_image_model__factory(likelihood)

    @genjax.gen
    def uniformpose_meshes_to_image_model(vertices_O, faces, vertex_colors):
        X_WC = uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ "camera_pose"

        N = vertices_O.shape[0]
        Xs_WO = (
            uniform_pose.vmap(in_axes=(0, 0))(
                jnp.ones((N, 3)) * -10.0, jnp.ones((N, 3)) * 10.0
            )
            @ "poses"
        )

        (observed_image, likelihood_metadata) = (
            meshes_to_image_model(X_WC, Xs_WO, vertices_O, faces, vertex_colors)
            @ "observed_image"
        )
        return (observed_image, likelihood_metadata)

    return uniformpose_meshes_to_image_model


def meshes_to_image_model__factory(likelihood):
    """
    This factory returns a generative function with one address, "observed_image", which
    generates an image of a collection of meshes with given poses, from a given camera pose.

    Factory arguments:
    - renderer: b3d.Renderer object
    - likelihood
        Should be a generative function with one random choice, "obs", of a random image.
        Should accept (vertices, faces, vertex_colors) as input.
        Should return (observed_image, metadata).
    - renderer_hyperparams: last argument for the likelihood
    """

    @genjax.gen
    def meshes_to_image_model(X_WC, Xs_WO, vertices_O, faces, vertex_colors):
        #                        (N, V, 3)    (N, F, 3)   (N, V, 3)
        #                        where N = number of objects
        # Coordinate frames:
        # W = world frame; C = camera frame; O = object frame

        vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(
            Xs_WO, vertices_O
        )
        vertices_C = X_WC.inv().apply(vertices_W.reshape(-1, 3))

        v = vertices_C.reshape(-1, 3)
        vc = vertex_colors.reshape(-1, 3)

        # shift up each object's face indices by the number of vertices in the previous objects
        N = vertices_O.shape[0]
        f = jax.vmap(lambda i, f: f + i * vertices_O.shape[1], in_axes=(0, 0))(
            jnp.arange(N), faces
        )
        f = f.reshape(-1, 3)

        observed_image, likelihood_metadata = likelihood(v, f, vc) @ "observed_image"

        return (observed_image, likelihood_metadata)

    return meshes_to_image_model


### Visualization code
def rr_log_uniformpose_meshes_to_image_model_trace(trace, renderer, **kwargs):
    """
    Log to rerun a visualization of a trace from `uniformpose_meshes_to_image_model`.
    """
    return rr_log_meshes_to_image_model_trace(
        trace,
        renderer,
        **kwargs,
        model_args_to_densemodel_args=(
            lambda args: (
                trace.get_choices()["camera_pose"],
                trace.get_choices()("poses").c.v,
                *args,
            )
        ),
    )


def rr_log_meshes_to_image_model_trace(
    trace,
    renderer,
    prefix="trace",
    timeless=False,
    model_args_to_densemodel_args=(lambda x: x),
    transform_Viz_Trace=Pose.identity(),
):
    """
    Log to rerun a visualization of a trace from `meshes_to_image_model`.

    The optional argument `model_args_to_densemodel_args` can be used to enable this function
    to visualize traces from other models that have the same return value as `meshes_to_image_model`.
    This function will call `model_args_to_densemodel_args` on the arguments of the given trace,
    and should produce arguments of the form accepted by `meshes_to_image_model`.

    The argument `transform_Viz_Trace` can be used to visualize the trace at a transformed
    coordinate frame.  `transform_Viz_Trace` is a Pose object so that for a 3D point
    `point_Trace` in the trace, `transform_Viz_Trace.apply(point_Trace)` is the corresponding
    3D point in the visualizer.
    """
    # 2D:
    (observed_rgbd, metadata) = trace.get_retval()
    rr.log(
        f"/{prefix}/rgb/observed",
        rr.Image(np.array(observed_rgbd[:, :, :3])),
        timeless=timeless,
    )
    rr.log(
        f"/{prefix}/depth/observed",
        rr.DepthImage(np.array(observed_rgbd[:, :, 3])),
        timeless=timeless,
    )

    # Visualization path for the average render,
    # if the likelihood metadata contains the output of the differentiable renderer.
    if "diffrend_output" in metadata:
        weights, attributes = metadata["diffrend_output"]
        avg_obs = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
        avg_obs_rgb_clipped = jnp.clip(avg_obs[:, :, :3], 0, 1)
        avg_obs_depth_clipped = jnp.clip(avg_obs[:, :, 3], 0, 1)
        rr.log(
            f"/{prefix}/rgb/average_render",
            rr.Image(np.array(avg_obs_rgb_clipped)),
            timeless=timeless,
        )
        rr.log(
            f"/{prefix}/depth/average_render",
            rr.DepthImage(np.array(avg_obs_depth_clipped)),
            timeless=timeless,
        )

    # 3D:
    rr.log(
        f"/{prefix}/3D/",
        rr.Transform3D(
            translation=transform_Viz_Trace.pos,
            mat3x3=transform_Viz_Trace.rot.as_matrix(),
        ),
        timeless=timeless,
    )

    (X_WC, Xs_WO, vertices_O, faces, vertex_colors) = model_args_to_densemodel_args(
        trace.get_args()
    )
    vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(
        Xs_WO, vertices_O
    )
    N = vertices_O.shape[0]
    f = jax.vmap(lambda i, f: f + i * vertices_O.shape[1], in_axes=(0, 0))(
        jnp.arange(N), faces
    )
    f = f.reshape(-1, 3)

    rr.log(
        f"/{prefix}/3D/mesh",
        rr.Mesh3D(
            vertex_positions=np.array(vertices_W.reshape(-1, 3)),
            triangle_indices=np.array(f),
            vertex_colors=np.array(vertex_colors.reshape(-1, 3)),
        ),
        timeless=timeless,
    )

    rr.log(
        f"/{prefix}/3D/camera",
        rr.Pinhole(
            focal_length=[float(renderer.fx), float(renderer.fy)],
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
        ),
        timeless=timeless,
    )
    rr.log(
        f"/{prefix}/3D/camera",
        rr.Transform3D(translation=X_WC.pos, mat3x3=X_WC.rot.as_matrix()),
        timeless=timeless,
    )
    xyzs_C = utils.xyz_from_depth(
        observed_rgbd[:, :, 3], renderer.fx, renderer.fy, renderer.cx, renderer.cy
    )
    xyzs_W = X_WC.apply(xyzs_C)
    rr.log(
        f"/{prefix}/3D/gt_pointcloud",
        rr.Points3D(
            positions=np.array(xyzs_W.reshape(-1, 3)),
            colors=np.array(observed_rgbd[:, :, :3].reshape(-1, 3)),
            radii=0.001 * np.ones(xyzs_W.reshape(-1, 3).shape[0]),
        ),
        timeless=timeless,
    )

    patch_centers_W = jax.vmap(lambda X_WO: X_WO.pos)(Xs_WO)
    rr.log(
        f"/{prefix}/3D/patch_centers",
        rr.Points3D(
            positions=np.array(patch_centers_W),
            colors=np.array([0.0, 0.0, 1.0]),
            radii=0.003,
        ),
        timeless=timeless,
    )
