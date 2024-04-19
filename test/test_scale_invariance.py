import b3d
import os
import jax.numpy as jnp
import rerun as rr

PORT = 8812
rr.init("real")
rr.connect(addr=f'127.0.0.1:{PORT}')


import trimesh
mesh_path = b3d.get_root_path() / "assets/objs/bunny.obj"
mesh = trimesh.load(mesh_path)

mesh_path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/025_mug/textured.obj")
mesh = trimesh.load(mesh_path)
mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)

width=200
height=200
fx=200.0
fy=200.0
cx=100.0
cy=100.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)

near_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.3, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0])
).inv()

far_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.9, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0])
).inv()

rgb_near, depth_near = renderer.render_attribute(
    near_pose[None,...],
    object_library.vertices,
    object_library.faces,
    object_library.ranges,
    object_library.attributes
)

rgb_far, depth_far = renderer.render_attribute(
    far_pose[None,...],
    object_library.vertices,
    object_library.faces,
    object_library.ranges,
    object_library.attributes
)



color_error, depth_error = (50.0, 0.01)
inlier_score, outlier_prob = (10.0, 0.0000001)
color_multiplier, depth_multiplier = (1.0, 1.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,

    inlier_score,
    outlier_prob,

    color_multiplier,
    depth_multiplier,
)

print(b3d.rgbd_sensor_model.logpdf(
    (rgb_near, depth_near), rgb_near, depth_near, model_args
))
print(b3d.rgbd_sensor_model.logpdf(
    (rgb_far, depth_far), rgb_far, depth_far, model_args
))


from genjax.generative_functions.distributions import ExactDensity
import genjax
class RGBDSensorModelWithRays(ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, model_args):
        return (rendered_rgb, rendered_depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, model_args):
        observed_rgb, observed_depth = observed
        inliers, _, _, valid_data_mask = b3d.get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args)
        
        inlier_score = model_args.inlier_score
        outlier_prob = model_args.outlier_prob
        multiplier = model_args.color_multiplier

        num_data_points = jnp.size(inliers)
        num_inliers = jnp.sum(inliers)
        num_no_data = jnp.sum(1.0 - valid_data_mask)
        num_outliers = num_data_points - num_inliers - num_no_data

        areas = (rendered_depth / fx)**2        

        p_in = (1.0 - outlier_prob) * inlier_score + outlier_prob
        p_out = outlier_prob

        scores = inliers * p_in + (1.0 - inliers) * p_out
        scores_with_area_adjustment = scores * areas

        sum_of_prob = jnp.sum(
            scores_with_area_adjustment
        )
        return jnp.log(sum_of_prob) * multiplier

rgbd_sensor_model_with_rays = RGBDSensorModelWithRays()

print(rgbd_sensor_model_with_rays.logpdf(
    (rgb_near, depth_near), rgb_near, depth_near, model_args
))
print(rgbd_sensor_model_with_rays.logpdf(
    (rgb_far, depth_far), rgb_far, depth_far, model_args
))
print(rgbd_sensor_model_with_rays.logpdf(
    (rgb_far, depth_far), rgb_near, depth_near, model_args
))
print(rgbd_sensor_model_with_rays.logpdf(
    (rgb_near, depth_near), rgb_far, depth_far, model_args
))






