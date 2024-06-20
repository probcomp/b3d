import genjax
import jax
import jax.numpy as jnp
from b3d.model import uniform_pose
import b3d.utils as utils
import b3d.differentiable_renderer as rendering
import rerun as rr

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

    @genjax.static_gen_fn
    def uniformpose_meshes_to_image_model(vertices_O, faces, vertex_colors):
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ "camera_pose"

        N = vertices_O.shape[0]
        Xs_WO = genjax.map_combinator(in_axes=(0, 0))(uniform_pose)(
            jnp.ones((N, 3))*-10.0, jnp.ones((N, 3))*10.0
        ) @ "poses"

        (observed_image, likelihood_metadata) = meshes_to_image_model(X_WC, Xs_WO, vertices_O, faces, vertex_colors) @ "observed_image"
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
    @genjax.static_gen_fn
    def meshes_to_image_model(X_WC, Xs_WO, vertices_O, faces, vertex_colors):
        #                        (N, V, 3)    (N, F, 3)   (N, V, 3)
        #                        where N = number of objects
        # Coordinate frames:
        # W = world frame; C = camera frame; O = object frame

        vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W.reshape(-1, 3))
        
        v = vertices_C.reshape(-1, 3)
        vc = vertex_colors.reshape(-1, 3)

        # shift up each object's face indices by the number of vertices in the previous objects
        N = vertices_O.shape[0]
        f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
        f = f.reshape(-1, 3)

        observed_image, likelihood_metadata = likelihood(v, f, vc) @ "observed_image"

        return (observed_image, likelihood_metadata)

    return meshes_to_image_model

### Visualization code
def rr_log_uniformpose_meshes_to_image_model_trace(trace, renderer):
    """
    Log to rerun a visualization of a trace from `uniformpose_meshes_to_image_model`.
    """
    return rr_log_meshes_to_image_model_trace(trace, renderer, model_args_to_densemodel_args=(
        lambda args: (trace["camera_pose"], trace["poses"], *args)
    ))

def rr_log_meshes_to_image_model_trace(
        trace, renderer,
        model_args_to_densemodel_args=(lambda x: x)
    ):
    """
    Log to rerun a visualization of a trace from `meshes_to_image_model`.

    The optional argument `model_args_to_densemodel_args` can be used to enable this function
    to visualize traces from other models that have the same return value as `meshes_to_image_model`.
    This function will call `model_args_to_densemodel_args` on the arguments of the given trace,
    and should produce arguments of the form accepted by `meshes_to_image_model`.
    """
    # 2D:
    (observed_rgbd, metadata) = trace.get_retval()
    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    
    # Visualization path for the average render,
    # if the likelihood metadata contains the output of the differentiable renderer.
    if "diffrend_output" in metadata:
        weights, attributes = metadata["diffrend_output"]
        avg_obs = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
        rr.log("/trace/rgb/average_render", rr.Image(avg_obs[:, :, :3]))
        rr.log("/trace/depth/average_render", rr.DepthImage(avg_obs[:, :, 3]))

    # 3D:
    (X_WC, Xs_WO, vertices_O, faces, vertex_colors) = model_args_to_densemodel_args(trace.get_args())
    Xs_WO = trace.strip()["poses"].inner.value # TODO: do this better
    vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
    N = vertices_O.shape[0]
    f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
    f = f.reshape(-1, 3)
    
    rr.log("/3D/trace/mesh", rr.Mesh3D(
        vertex_positions=vertices_W.reshape(-1, 3),
        indices=f,
        vertex_colors=vertex_colors.reshape(-1, 3)
    ))

    rr.log("/trace/camera",
        rr.Pinhole(
            focal_length=renderer.fx,
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
            )
        )
    rr.log("/trace/camera", rr.Transform3D(translation=X_WC.pos, mat3x3=X_WC.rot.as_matrix()))
    xyzs_C = utils.xyz_from_depth(observed_rgbd[:, :, 3], renderer.fx, renderer.fy, renderer.cx, renderer.cy)
    xyzs_W = X_WC.apply(xyzs_C)
    rr.log("/3D/trace/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )
