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
from collections import namedtuple

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

ModelArgs = namedtuple('ModelArgs', [
    'color_tolerance',
    'depth_tolerance',
    'inlier_score',
    'outlier_prob',
    'color_multiplier',
    'depth_multiplier',
])

def get_rgb_depth_inliers_from_trace(trace):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    model_args = trace.get_args()[1]
    return get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args)

def get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args):
    observed_lab = b3d.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.rgb_to_lab(rendered_rgb)

    valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

    error = (
        jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) + 
        jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
    )
    color_inliers = (error < model_args.color_tolerance) * valid_data_mask
    depth_inliers = (jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance) * valid_data_mask
    inliers = color_inliers * depth_inliers * valid_data_mask
    return (inliers, color_inliers, depth_inliers, valid_data_mask)

class RGBDSensorModel(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, model_args):
        return (rendered_rgb, rendered_depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, model_args):
        observed_rgb, observed_depth = observed
        inliers, _, _, valid_data_mask = get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args)
        
        inlier_score = model_args.inlier_score
        outlier_prob = model_args.outlier_prob
        multiplier = model_args.color_multiplier

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




def model_multiobject_gl_factory(renderer, image_likelihood=rgbd_sensor_model):
    @genjax.static_gen_fn
    def model(
        _num_obj_arr, # new 
        model_args,
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
        observed_rgb, observed_depth = image_likelihood(
            rendered_rgb, rendered_depth,
            model_args
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




def rerun_visualize_trace_t(trace, t, modes=["rgb", "depth", "inliers"]):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    rr.set_time_sequence("frame", t)

    if "rgb" in modes:
        rr.log("/image", rr.Image(observed_rgb))
        rr.log("/image/rgb_rendering", rr.Image(rendered_rgb))

    if "depth" in modes:
        rr.log("/image/depth", rr.DepthImage(observed_depth))
        rr.log("/image/depth_rendering", rr.DepthImage(rendered_depth))

    info_string = f"# Score : {trace.get_score()}"

    if "inliers" in modes:
        (inliers, color_inliers, depth_inliers, valid_data_mask) = b3d.get_rgb_depth_inliers_from_trace(trace)
        rr.log("/image/color_inliers", rr.DepthImage(color_inliers * 1.0))
        rr.log("/image/depth_inliers", rr.DepthImage(depth_inliers * 1.0))
        info_string += f"\n # Inliers : {jnp.sum(inliers)}"
        info_string += f"\n # Valid : {jnp.sum(valid_data_mask)}"
    rr.log("/info", rr.TextDocument(info_string))
