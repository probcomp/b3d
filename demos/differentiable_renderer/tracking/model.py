import jax
import jax.numpy as jnp
import genjax
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as rendering
import b3d
from b3d import Pose
from b3d.model import uniform_pose
import rerun as rr

def normalize(v):
    return v / jnp.sum(v)

def model_singleobject_gl_factory(renderer,
                                  lab_noise_scale=3.0,
                                  depth_noise_scale=0.07,
                                  outlier_prob = 0.05,
                                  hyperparams = rendering.DEFAULT_HYPERPARAMS
                                ):
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors):
        X_WO = uniform_pose(jnp.ones(3)*-10.0, jnp.ones(3)*10.0) @ "pose"
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
        
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, hyperparams
        )
        weights = weights.at[:, :, 0].set(weights[:, :, 0] + outlier_prob)
        weights = jax.vmap(normalize, in_axes=(0,))(weights.reshape(-1, weights.shape[-1])).reshape(weights.shape)

        observed_rgbd = likelihoods.mixture_rgbd_sensor_model(
            weights, attributes,
            lab_noise_scale, depth_noise_scale, 0., 10.
        ) @ "observed_rgbd"
        average_rgbd = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
        return (observed_rgbd, average_rgbd)
    return model

def rr_viz_trace(trace, renderer):
    (observed_rgbd, average_rgbd) = trace.get_retval()
    # rr.log("/trace/vertices", rr.Points3D(positions=vertices, colors=jnp.ones_like(vertices), radii=0.001))
    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/rgb/rendered_average", rr.Image(average_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    rr.log("/trace/depth/rendered_average", rr.DepthImage(average_rgbd[:, :, 3]))
    
    (vertices, faces, vertex_colors) = trace.get_args()
    pose = trace["pose"]
    cam_pose = trace["camera_pose"]
    rr.log("/trace/mesh", rr.Mesh3D(
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