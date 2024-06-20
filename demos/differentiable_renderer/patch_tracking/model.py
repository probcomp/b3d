import jax
import jax.numpy as jnp
import genjax
import b3d.chisight.dense.differentiable_renderer as rendering
import b3d
from b3d import Pose
from b3d.model import uniform_pose
import rerun as rr
import demos.differentiable_renderer.patch_tracking.demo_utils as utils

def normalize(v):
    return v / jnp.sum(v)

### Single object model ###

def single_object_model_factory(
        renderer, likelihood,
        renderer_hyperparams
    ):
    """
    Args:
    - renderer
    - likelihood
        Should be a distribution on images.
        Should accept (weights, attributes, *likelihood_args) as input.
    - renderer_hyperparams
    """
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors, likelihood_args):
        X_WO = uniform_pose(jnp.ones(3)*-10.0, jnp.ones(3)*10.0) @ "pose"
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
        
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, renderer_hyperparams
        )

        observed_rgbd = likelihood(weights, attributes) @ "observed_rgbd"
        return (observed_rgbd, weights, attributes)
    return model

def rr_log_trace(trace, renderer, prefix="trace"):
    # 2D:
    (observed_rgbd, weights, attributes) = trace.get_retval()
    rr.log(f"/{prefix}/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log(f"/{prefix}/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    avg_obs = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
    rr.log(f"/{prefix}/rgb/average_render", rr.Image(avg_obs[:, :, :3]))
    rr.log(f"/{prefix}/depth/average_render", rr.DepthImage(avg_obs[:, :, 3]))

    # 3D:
    (vertices, faces, vertex_colors, _) = trace.get_args()
    pose = trace["pose"]
    cam_pose = trace["camera_pose"]
    
    rr.log(f"3D/{prefix}/mesh", rr.Mesh3D(
        vertex_positions=pose.apply(vertices),
        indices=faces,
        vertex_colors=vertex_colors
    ))

    rr.log("3D/{prefix}/camera",
        rr.Pinhole(
            focal_length=renderer.fx,
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
            )
        )
    rr.log("3D/{prefix}/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()))
    xyzs_C = utils.unproject_depth(observed_rgbd[:, :, 3], renderer)
    xyzs_W = cam_pose.apply(xyzs_C)
    rr.log("/3D/{prefix}/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )

### Multiple object model ###
def multiple_object_model_factory(
        renderer, likelihood,
        renderer_hyperparams
    ):
    """
    Args:
    - renderer
    - likelihood
        Should be a distribution on images.
        Should accept (weights, attributes, *likelihood_args) as input.
    - renderer_hyperparams
    """
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors, likelihood_args):
        #    (N, V, 3)    (N, F, 3)   (N, V, 3)
        # where N = num objects
        N = vertices_O.shape[0]

        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"

        Xs_WO = genjax.map_combinator(in_axes=(0, 0))(uniform_pose)(
            jnp.ones((N, 3))*-10.0, jnp.ones((N, 3))*10.0
        ) @ "poses"
        
        vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W.reshape(-1, 3))
        
        v = vertices_C.reshape(-1, 3)
        vc = vertex_colors.reshape(-1, 3)

        # shift up each object's face indices by the number of vertices in the previous objects
        f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
        f = f.reshape(-1, 3)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, v, f, vc, renderer_hyperparams
        )

        observed_rgbd = likelihood(weights, attributes) @ "observed_rgbd"
        return (observed_rgbd, weights, attributes)
    return model

def rr_log_multiobject_trace(trace, renderer):
    # 2D:
    (observed_rgbd, weights, attributes) = trace.get_retval()
    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    avg_obs = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
    rr.log("/trace/rgb/average_render", rr.Image(avg_obs[:, :, :3]))
    rr.log("/trace/depth/average_render", rr.DepthImage(avg_obs[:, :, 3]))

    # 3D:
    (vertices_O, faces, vertex_colors, _) = trace.get_args()
    Xs_WO = trace.strip()["poses"].inner.value # TODO: do this better
    vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
    N = vertices_O.shape[0]
    f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
    f = f.reshape(-1, 3)

    cam_pose = trace["camera_pose"]
    
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
    rr.log("/trace/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()))
    xyzs_C = utils.unproject_depth(observed_rgbd[:, :, 3], renderer)
    xyzs_W = cam_pose.apply(xyzs_C)
    rr.log("/3D/trace/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )
