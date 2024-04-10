"""
Copy of the model where I'm trying to add a prior over object meshes.
"""

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
import b3d.utils as utils

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

@genjax.static_gen_fn
def generate_object_voxel_mesh(
    img_width, img_height, subx, suby,
    intrinsics,
    p_occupancy
):
    """
        img_width = image width
        img_height = image height
        subx = amount to subsample along width (have img_width/subx gradations)
        suby = amount to subsample along height (have img_height/suby gradations)
    """
    img_width = img_width.const
    img_height = img_height.const
    subx = subx.const
    suby = suby.const
    intrinsics = intrinsics.const

    (fx,fy, cx,cy,near,far) = intrinsics
    # print(f"near = {near}, far = {far}")

    w = img_width // subx
    h = img_height // suby

    # Decide which "pixels" will have a visible voxel there
    voxel_present = genjax.map_combinator(in_axes=(0,))(
        genjax.map_combinator(in_axes=(0,))(genjax.flip)
    )(p_occupancy * jnp.ones((w, h))) @ "voxel_present"

    depth = genjax.map_combinator(in_axes=(0, (0, 0)))(
        genjax.map_combinator(in_axes=(0, (0, 0)))(
            genjax.masking_combinator(genjax.uniform)
        )
    )(voxel_present, (near * jnp.ones((w, h)), far * jnp.ones((w, h)))) @ "depth"
    
    colors = genjax.map_combinator(in_axes=(0, (0, 0)))(    # width
        genjax.map_combinator(in_axes=(0, (0, 0)))(         # height
            genjax.map_combinator(in_axes=(None, (0, 0)))(  # color
                genjax.masking_combinator(genjax.uniform)
            )
        )
    )(voxel_present, (jnp.zeros((w, h, 3)), jnp.ones((w, h, 3)))) @ "color"

    ##  set color & depth values to -1 for invalid slts
    # function which takes a mask object on a single number
    # and returns -1.0 if the mask object is invalid, otherwise
    matcher = lambda mask_object: mask_object.match(lambda: -1.0, lambda x : x)
    # map this function over the depth array & color array
    depth = utils.multivmap_at_zero(matcher, 2)(depth)
    colors = utils.multivmap_at_zero(matcher, 3)(colors)

    voxel_present_flat = voxel_present.reshape(-1)
    depth_flat = depth.reshape(-1)
    colors_flat = colors.reshape(-1, 3)
    resolutions = depth_flat / fx * 2.0 * jnp.sqrt(subx * suby) * 1.25

    i2 = (fx // subx, fy // suby, cx // subx, cy // suby, near, far)
    point_centers = utils.unproject_depth(depth, i2).reshape(-1, 3)

    # return (voxel_present, depth, colors, resolutions, point_centers)

    vertices, faces, vertex_colors, face_colors = utils.make_mesh_from_point_cloud_and_resolution_2(
        point_centers,
        colors_flat,
        resolutions,
        voxel_present_flat
    )

    return vertices, faces, vertex_colors

@genjax.static_gen_fn
def generate_voxel_object_library(n_objects, object_model_args):
    object_library = b3d.MeshLibrary.make_empty_library()
    for i in range(n_objects):
        mesh = generate_object_voxel_mesh(*object_model_args) @ ("mesh", i)
        object_library.add_object(*mesh)
    return object_library

@genjax.static_gen_fn
def step_model(state):
    (camera_pose, object_poses) = state
    new_camera_pose = gaussian_vmf_pose(camera_pose, 0.05, 0.05) @ "camera_pose"
    
    # new_poses_as_mtx = jnp.empty((0,4,4))
    new_poses = []
    for i in range(len(object_poses)):
        prev_pose = object_poses[i] # Pose.from_matrix(object_poses_as_mtx[i])
        object_pose = gaussian_vmf_pose(prev_pose, 0.05, 0.05) @ (i, "object_pose")
        # new_poses_as_mtx = jnp.concatenate([new_poses_as_mtx, object_pose.as_matrix()[None, ...]])
        new_poses.append(object_pose)

    return (new_camera_pose, new_poses)

def model_multiobject_gl_factory(renderer):
    @genjax.static_gen_fn
    def obs_model(poses, camera_pose, 
                object_library, library_obj_indices_to_render,
                color_error, depth_error, inlier_score, outlier_prob, color_multiplier, depth_multiplier
                ):
        poses_as_mtx = jnp.stack([(camera_pose.inv() @ pose).as_matrix() for pose in poses])

        # poses_as_mtx = camera_pose.inv() @ poses_as_mtx
        rendered_rgb, rendered_depth = renderer.render_attribute(
            poses_as_mtx,
            object_library.vertices,
            object_library.faces,
            object_library.ranges[library_obj_indices_to_render] * (library_obj_indices_to_render >= 0).reshape(-1,1),
            object_library.attributes
        )
        observed_rgb = rgb_sensor_model(
            rendered_rgb, color_error, inlier_score, outlier_prob, color_multiplier
        ) @ "observed_rgb"

        observed_depth = depth_sensor_model(
            rendered_depth, depth_error, inlier_score, outlier_prob, depth_multiplier
        ) @ "observed_depth"
        return (observed_rgb, rendered_rgb), (observed_depth, rendered_depth)

    @genjax.static_gen_fn
    def model(
        _num_obj_arr, # new 

        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier,

        object_model_args,

        _max_n_frames_arr,
        n_frames
    ):

        # object_library = generate_voxel_object_library(_num_obj_arr.shape[0], object_model_args) @ "object_library"
        object_library = b3d.MeshLibrary.make_empty_library()
        vertices, faces, vertex_colors = generate_object_voxel_mesh(*object_model_args) @ "mesh"
        object_library.add_object(vertices, faces, vertex_colors)

        # poses_as_mtx = jnp.empty((0,4,4))
        poses = []
        library_obj_indices_to_render = jnp.empty((0,), dtype=int)
        camera_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"

        for i in range(_num_obj_arr.shape[0]):        
            object_identity = uniform_discrete(jnp.arange(-1, len(object_library.ranges))) @ f"object_{i}"  # TODO possible_object_indices?
            library_obj_indices_to_render = jnp.concatenate((library_obj_indices_to_render, jnp.array([object_identity])))

            object_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"object_pose_{i}"
            poses.append(object_pose)

        obs_init = obs_model(
            poses, camera_pose,
            object_library, library_obj_indices_to_render,
            color_error, depth_error, inlier_score, outlier_prob, color_multiplier, depth_multiplier
        ) @ "obs"

        steps = genjax.unfold_combinator(
            max_length=_max_n_frames_arr.shape[0]
        )(step_model)(
            n_frames, (camera_pose, poses)
        ) @ "steps"

        obs_steps = genjax.map_combinator(
            in_axes=(0, (0, 0, None, None, None, None, None, None, None, None))
        )(
            genjax.masking_combinator(obs_model)
        )(
            jnp.arange(_max_n_frames_arr.shape[0]) < n_frames,
            (
                steps[1], steps[0],
                object_library, library_obj_indices_to_render,
                color_error, depth_error, inlier_score, outlier_prob, color_multiplier, depth_multiplier
            )
        ) @ "obs_steps"

        return object_library, obs_init, obs_steps

    return model


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
    object_library, (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    rr.set_time_sequence("frame", t)

    rr.log("/image", rr.Image(observed_rgb))
    rr.log("/image/rgb_rendering", rr.Image(rendered_rgb))

    rr.log("/image/depth", rr.DepthImage(observed_depth))
    rr.log("/image/depth_rendering", rr.DepthImage(rendered_depth))

    for i in range(object_library.get_num_objects()):
        vp, f, vc = object_library.get_object(i)
        rr.log(
            f"/objects_meshes/{i}",
            rr.Mesh3D(
                vertex_positions=vp,
                indices=f,
                vertex_colors=vc
            ),
            timeless=True
        )

def rerun_visualize_trace_across_time(trace):
    object_library, obs_init, obs_step = trace.get_retval()
    
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = obs_init
    rr.set_time_sequence("frame", 0)
    rr.log("/image", rr.Image(observed_rgb))
    rr.log("/image/rgb_rendering", rr.Image(rendered_rgb))
    rr.log("/image/depth", rr.DepthImage(observed_depth))
    rr.log("/image/depth_rendering", rr.DepthImage(rendered_depth))
    for i in range(object_library.get_num_objects()):
        vp, f, vc = object_library.get_object(i)
        rr.log(
            f"/objects_meshes/{i}",
            rr.Mesh3D(
                vertex_positions=vp,
                indices=f,
                vertex_colors=vc
            )
        )

    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = obs_step.value
    for j in range(observed_rgb.shape[0]):
        if obs_step.flag[j]:
            t = j + 1
            rr.set_time_sequence("frame", t)
            rr.log("/image", rr.Image(observed_rgb[j]))
            rr.log("/image/rgb_rendering", rr.Image(rendered_rgb[j]))
            rr.log("/image/depth", rr.DepthImage(observed_depth[j]))
            rr.log("/image/depth_rendering", rr.DepthImage(rendered_depth[j]))

            # TODO: revisualize objects at new poses

    # TODO: add visualizations for the camera