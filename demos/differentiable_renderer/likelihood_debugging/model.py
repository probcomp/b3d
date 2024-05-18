import jax
import jax.numpy as jnp
import genjax
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as rendering
import b3d
from b3d import Pose
from b3d.model import uniform_pose
import rerun as rr
import demos.differentiable_renderer.tracking.utils as utils

def normalize(v):
    return v / jnp.sum(v)

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
    - get_example_rgbd
        Should accept (weights, attributes, likelihood_args) as input
        and return an rgbd image.
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

def rr_log_trace(trace, renderer):
    # 2D:
    (observed_rgbd, weights, attributes) = trace.get_retval()
    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    avg_obs = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
    rr.log("/trace/rgb/average_render", rr.Image(avg_obs[:, :, :3]))
    rr.log("/trace/depth/average_render", rr.DepthImage(avg_obs[:, :, 3]))

    # 3D:
    (vertices, faces, vertex_colors, _) = trace.get_args()
    pose = trace["pose"]
    cam_pose = trace["camera_pose"]
    
    rr.log("3D/trace/mesh", rr.Mesh3D(
        vertex_positions=pose.apply(vertices),
        indices=faces,
        vertex_colors=vertex_colors
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
    # c = observed_rgbd[:, :, :3].reshape(-1,3)
    # ca = jnp.concatenate([c, alpha*jnp.ones((c.shape[0], 1))], axis=1)
    rr.log("/3D/trace/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )

