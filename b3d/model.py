import genjax
from b3d.pose import Pose, sample_uniform_pose, sample_gaussian_vmf_pose
from genjax.generative_functions.distributions import ExactDensity
import jax
import jax.numpy as jnp
import b3d
from jax.scipy.special import logsumexp
import rerun as rr

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




class DepthSensorModel(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_depth, tolerance, inlier_score, outlier_prob, multiplier):
        return rendered_depth

    def logpdf(self, observed_depth,
               rendered_depth, tolerance, inlier_score, outlier_prob, multiplier):

        valid_data_mask = (rendered_depth != 0.0)
        inlier_match_mask = (jnp.abs(observed_depth - rendered_depth) < tolerance)
        inlier_match_mask = inlier_match_mask * valid_data_mask

        logp_in = jnp.log((1.0 - outlier_prob) * inlier_score + outlier_prob)
        logp_out = jnp.log(outlier_prob)
        logp_no_data = jnp.log(1 / 1.0)

        num_data_points = jnp.size(inlier_match_mask)
        num_inliers = jnp.sum(inlier_match_mask)
        num_no_data = jnp.sum(1.0 - valid_data_mask)
        num_outliers = num_data_points - num_inliers - num_no_data

        log_sum_of_probs = logsumexp(jnp.array([
            jnp.log(num_inliers) + logp_in,
            jnp.log(num_outliers) + logp_out,
            jnp.log(num_no_data) + logp_no_data
        ]))
        average_log_prob = log_sum_of_probs - jnp.log(num_data_points)
        return average_log_prob * multiplier

depth_sensor_model = DepthSensorModel()

def color_error_helper(observed_rgb, rendered_rgb, lab_tolerance):
    valid_data_mask = (rendered_rgb.sum(-1) != 0.0)
    # valid_data_mask = jnp.full(valid_data_mask.shape, True)
    observed_lab = b3d.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.rgb_to_lab(rendered_rgb)
    error = (
        jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) + 
        jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
    )
    inlier_match_mask = (error < lab_tolerance)
    inlier_match_mask = inlier_match_mask * valid_data_mask

    num_data_points = jnp.size(inlier_match_mask)
    num_inliers = jnp.sum(inlier_match_mask)
    num_no_data = jnp.sum(1.0 - valid_data_mask)
    num_outliers = num_data_points - num_inliers - num_no_data
    return inlier_match_mask, num_data_points, num_inliers, num_no_data, num_outliers

class RGBSensorModel(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, lab_tolerance, inlier_score, outlier_prob, multiplier):
        return rendered_rgb

    def logpdf(self, observed_rgb,
               rendered_rgb, lab_tolerance, inlier_score, outlier_prob, multiplier):
        
        inlier_match_mask, num_data_points, num_inliers, num_no_data, num_outliers = color_error_helper(
            observed_rgb, rendered_rgb, lab_tolerance
        )
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

rgb_sensor_model = RGBSensorModel()



def model_gl_factory(renderer):
    @genjax.static_gen_fn
    def model(
        vertices,
        faces,
        colors,

        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier
    ):
        object_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ "object_pose"
        camera_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ "camera_pose"
        rendered_rgb, rendered_depth = renderer.render_attribute(
            (camera_pose.inv() @ object_pose).as_matrix()[None,...],
            vertices, faces, jnp.array([[0, len(faces)]], dtype=jnp.int32), colors
        )
        observed_rgb = rgb_sensor_model(
            rendered_rgb, color_error, inlier_score, outlier_prob, color_multiplier
        ) @ "observed_rgb"

        observed_depth = depth_sensor_model(
            rendered_depth, depth_error, inlier_score, outlier_prob, depth_multiplier
        ) @ "observed_depth"
        return (observed_rgb, rendered_rgb), (observed_depth, rendered_depth)
    return model


def rerun_visualize_trace_t(trace, t):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    (
        vertices, faces, vertex_colors,
        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier
    ) = trace.get_args()

    rr.set_time_sequence("frame", t)

    # rr.log("/get_score", rr.Scalar(trace.get_score()))
    rr.log("/rgb/image", rr.Image(observed_rgb))
    rr.log("/rgb/image/rendering", rr.Image(rendered_rgb))

    rr.log("/depth/image/", rr.DepthImage(observed_depth))
    rr.log("/depth/image/rendering", rr.DepthImage(rendered_depth))

    lab_tolerance = color_error

    inlier_match_mask, num_data_points, num_inliers, num_no_data, num_outliers = color_error_helper(
        observed_rgb, rendered_rgb, lab_tolerance
    )

    rr.log("/rgb/image/inliers", rr.Image(inlier_match_mask * 1.0))

    rr.log("text_document", 
        rr.TextDocument(f'''
# Score: {trace.get_score()} \n
# num_inliers: {num_inliers} \n
# num_no_data: {num_no_data} \n
# num_outliers: {num_outliers} \n
# inlier_score: {inlier_score} \n
# Outlier Prob: {outlier_prob} \n
# color multiplier: {color_multiplier} \n
# depth multiplier: {depth_multiplier} \n
'''.strip(),
            media_type=rr.MediaType.MARKDOWN
                        ))