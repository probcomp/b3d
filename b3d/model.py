import genjax
from b3d.pose import Pose, sample_uniform_pose, sample_gaussian_vmf_pose
from genjax.generative_functions.distributions import ExactDensity
import jax
import jax.numpy as jnp
import b3d
from jax.scipy.special import logsumexp
import rerun as rr
from collections import OrderedDict
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

class UniformDiscrete(ExactDensity, genjax.JAXGenerativeFunction):
    def sample(self, key, vals):
        return jax.random.choice(key, vals)

    def logpdf(self, sampled_val, vals, **kwargs):
        return jnp.log(1.0 / (vals.shape[0]))
uniform_discrete = UniformDiscrete()

class UniformPose(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, low, high):
        return sample_uniform_pose(key, low, high)

    def logpdf(self, pose, low, high):
        position = pose.pos
        valid = ((low <= position) & (position <= high))
        position_score = jnp.log((valid * 1.0) * (jnp.ones_like(position) / (high-low)))
        return position_score.sum() + jnp.pi**2

class GaussianPose(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, mean_pose, variance, concentration):
        return sample_gaussian_vmf_pose(key, mean_pose, variance, concentration)

    def logpdf(self, pose, mean_pose, variance, concentration):
        translation_score = tfp.distributions.MultivariateNormalDiag(
        mean_pose.pos, jnp.ones(3) * variance).log_prob(pose.pos)
        quaternion_score = tfp.distributions.VonMisesFisher(
            mean_pose.quat / jnp.linalg.norm(mean_pose.quat), concentration
        ).log_prob(pose.quat)
        return translation_score + quaternion_score

uniform_pose = UniformPose()
gaussian_vmf_pose = GaussianPose()


class RGBDSensorModel(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, color_tolerance, depth_tolerance, inlier_score, outlier_prob, multiplier):
        return (rendered_rgb, rendered_depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, color_tolerance, depth_tolerance, inlier_score, outlier_prob, multiplier):
        observed_rgb, observed_depth = observed

        observed_lab = b3d.rgb_to_lab(observed_rgb)
        rendered_lab = b3d.rgb_to_lab(rendered_rgb)

        valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

        error = (
            jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) + 
            jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
        )
        color_inliers = (error < color_tolerance)
        depth_inliers = (jnp.abs(observed_depth - rendered_depth) < depth_tolerance)
        inliers = color_inliers * depth_inliers * valid_data_mask

        num_data_points = jnp.size(inliers)
        num_inliers = jnp.sum(inliers)
        num_no_data = jnp.sum(1.0 - valid_data_mask)
        num_outliers = num_data_points - num_inliers - num_no_data
        
        logp_in = jnp.log((1.0 - outlier_prob) * inlier_score + outlier_prob)
        logp_out = jnp.log(outlier_prob)
        logp_no_data = jnp.log(1 / 1.0)

        log_sum_of_probs = logsumexp(jnp.array([
            jnp.log(num_inliers) + logp_in,
            jnp.log(num_outliers) + logp_out,
            jnp.log(num_no_data) + logp_no_data
        ]))
        average_log_prob = log_sum_of_probs - jnp.log(num_data_points)
        return average_log_prob * multiplier

rgbd_sensor_model = RGBDSensorModel()

def model_multiobject_gl_factory(renderer):
    @genjax.static_gen_fn
    def model(
        _num_obj_arr, # new 

        color_tolerance,
        depth_tolerance,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier,
        object_library
    ):

        object_poses = Pose(jnp.zeros((0,3)), jnp.zeros((0,4)))
        object_indices = jnp.empty((0,), dtype=int)
        camera_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
            
        for i in range(_num_obj_arr.shape[0]):        
            object_identity = uniform_discrete(jnp.arange(-1, len(object_library.ranges))) @ f"object_{i}"
            object_indices = jnp.concatenate((object_indices, jnp.array([object_identity])))

            object_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"object_pose_{i}"
            object_poses = Pose.concatenate_poses([object_poses, camera_pose.inv() @ object_pose[None,...]])

        rendered_rgb, rendered_depth = renderer.render_attribute(
            object_poses,
            object_library.vertices,
            object_library.faces,
            object_library.ranges[object_indices] * (object_indices >= 0).reshape(-1,1),
            object_library.attributes
        )
        observed_rgb, observed_depth = rgbd_sensor_model(
            rendered_rgb, rendered_depth, color_tolerance, depth_tolerance, inlier_score, outlier_prob, color_multiplier
        ) @ "observed_rgb_depth"
        return (observed_rgb, rendered_rgb), (observed_depth, rendered_depth)
    return model

def get_rendered_rgb_depth_from_trace(trace):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval
    return (rendered_rgb, rendered_depth)

def get_poses_from_trace(trace):
    return Pose.stack_poses([
        trace[f"object_pose_{i}"] for i in range(len(trace.get_args()[0]))
    ])

def get_object_ids_from_trace(trace):
    return jnp.array([
        trace[f"object_{i}"] for i in range(len(trace.get_args()[0]))
    ])
    
def get_rgb_inlier_outlier_from_trace(trace):
    lab_tolerance = trace.get_args()[1]
    observed_rgb, rendered_rgb = trace.get_retval()[0]
    inlier_match_mask = color_error_helper(
        observed_rgb, rendered_rgb, lab_tolerance
    )[0]
    
    return (inlier_match_mask, 1 - inlier_match_mask)
    
def get_depth_inlier_outlier_from_trace(trace):
    depth_tolerance = trace.get_args()[2]
    observed_depth, rendered_depth = trace.get_retval()[1]
    valid_data_mask = (rendered_depth != 0.0)
    inlier_match_mask = (jnp.abs(observed_depth - rendered_depth) < depth_tolerance)
    inlier_match_mask = inlier_match_mask * valid_data_mask

    return (inlier_match_mask, 1 - inlier_match_mask)

def get_rgb_depth_inlier_outlier_from_trace(trace):
    rgb_inlier_mask = get_rgb_inlier_outlier_from_trace(trace)[0]
    depth_inlier_mask = get_depth_inlier_outlier_from_trace(trace)[0]
    
    rgb_and_depth_inlier_mask = rgb_inlier_mask * depth_inlier_mask
    
    return (rgb_and_depth_inlier_mask, 1 - rgb_and_depth_inlier_mask)

def rerun_visualize_trace_t(trace, t):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    rr.set_time_sequence("frame", t)

    rr.log("/image", rr.Image(observed_rgb))
    rr.log("/image/rgb_rendering", rr.Image(rendered_rgb))

    rr.log("/image/depth", rr.DepthImage(observed_depth))
    rr.log("/image/depth_rendering", rr.DepthImage(rendered_depth))
