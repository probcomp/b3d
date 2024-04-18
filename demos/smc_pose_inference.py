import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm
import trimesh

PORT = 8812
rr.init("real")
rr.connect(addr=f'127.0.0.1:{PORT}')

video_input = b3d.VideoInput.load(os.path.join(b3d.get_root_path(),
# "assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz"
"assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
))

scaling_factor = 5
image_width, image_height, fx,fy, cx,cy,near,far = jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

_rgb = video_input.rgb[0].astype(jnp.float32) / 255.0
_depth = video_input.xyz[0].astype(jnp.float32)[...,2]
rgb = jnp.clip(jax.image.resize(
    _rgb, (image_height, image_width, 3), "nearest"
), 0.0, 1.0)
depth = jax.image.resize(
    _depth, (image_height, image_width), "nearest"
)

point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1,3)
rr.log("point_cloud", rr.Points3D(point_cloud.reshape(-1,3), colors=rgb.reshape(-1,3)))
table_pose, table_dims = b3d.Pose.fit_table_plane(point_cloud, 0.01, 0.01, 100, 1000)
b3d.rr_log_pose("table", table_pose)

mesh_path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/025_mug/textured.obj")
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = (jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0 ) 

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)



rgb_object_samples = vertex_colors[jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))]
distances = jnp.abs(rgb[...,None] - rgb_object_samples.T).sum([-1,-2])
rr.log("image/distances", rr.DepthImage(distances))

object_center_hypothesis = point_cloud[distances.argmin()]




from functools import partial
@partial(jax.jit, static_argnames=['address', 'number'])
def gvmf_pose_proposal(trace, key, variance, concentration, address, number):
    addr = address.const
    test_poses = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
        jax.random.split(key, number),
        trace[addr],
        variance, concentration
    )])
    scores = b3d.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), genjax.Pytree.const([addr]), test_poses
    )
    sample = jax.random.categorical(key, scores)
    trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const([addr]),
            test_poses[scores.argmax()])
    key = jax.random.split(key, 2)[-1]
    return trace, key


color_error, depth_error = (50.0, 0.01)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (3000.0, 3000.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,

    inlier_score,
    outlier_prob,

    color_multiplier,
    depth_multiplier,
)

key = jax.random.PRNGKey(1000)


# from genjax.generative_functions.distributions import ExactDensity
# class RGBDSensorModel(ExactDensity,genjax.JAXGenerativeFunction):
#     def sample(self, key, rendered_rgb, rendered_depth, model_args):
#         return (rendered_rgb, rendered_depth)

#     def logpdf(self, observed, rendered_rgb, rendered_depth, model_args):
#         observed_rgb, observed_depth = observed
#         observed_lab = b3d.rgb_to_lab(observed_rgb)
#         rendered_lab = b3d.rgb_to_lab(rendered_rgb)

#         inlier_score = model_args.inlier_score
#         outlier_prob = model_args.outlier_prob
#         multiplier = model_args.color_multiplier

#         valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

#         error = (
#             jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) + 
#             jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
#         )
#         color_inliers = (error < model_args.color_tolerance) * valid_data_mask
#         depth_inliers = (jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance) * valid_data_mask
#         inliers = color_inliers * depth_inliers

#         num_data_points = jnp.size(inliers)
#         num_inliers = jnp.sum(inliers)
#         num_outliers = num_data_points - num_inliers

#         logp_in = jnp.log((1.0 - outlier_prob) * inlier_score + outlier_prob)
#         logp_out = jnp.log(outlier_prob)

#         log_sum_of_probs = b3d.logsumexp(jnp.array([
#             jnp.log(num_inliers) + logp_in,
#             jnp.log(num_outliers) + logp_out,
#         ]))
#         average_log_prob = log_sum_of_probs - jnp.log(num_data_points)
#         return average_log_prob * multiplier

# rgbd_sensor_model = RGBDSensorModel()

model = b3d.model_multiobject_gl_factory(renderer, b3d.rgbd_sensor_model)

importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(0)


key = jax.random.split(key, 2)[-1]
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.sample_gaussian_vmf_pose(
                key, Pose.from_translation(object_center_hypothesis), 0.001, 0.01),
            "object_0": 0,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    (jnp.arange(1), model_args, object_library)
)
b3d.rerun_visualize_trace_t(trace, 0)

# key = jax.random.split(key, 2)[-1]
# test_poses = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
#     jax.random.split(key, 300000),
#     trace["object_pose_0"],
#     0.03, 0.0001
# )
# test_poses_batches = test_poses.split(300)
# scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(trace, key, genjax.Pytree.const(["object_pose_0"]), poses) for poses in test_poses_batches])
# print(scores.max())
# trace = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_pose_0"]),  test_poses[scores.argmax()])
# print(trace.get_score())
# b3d.rerun_visualize_trace_t(trace, 0)

for i in range(10):
    key = jax.random.split(key, 2)[-1]
    test_poses = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
        jax.random.split(key, 10000),
        trace["object_pose_0"],
        0.01, 1000.0
    )
    test_poses_batches = test_poses.split(10)
    scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(trace, key, genjax.Pytree.const(["object_pose_0"]), poses) for poses in test_poses_batches])
    if scores.max() > trace.get_score():
        trace = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_pose_0"]),  test_poses[scores.argmax()])
        b3d.rerun_visualize_trace_t(trace, 0)
    else:
        print(f"skip {i}")



delta_cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.02, 0.02, 51),
        jnp.linspace(-0.02, 0.02, 51),
        jnp.linspace(-jnp.pi, jnp.pi, 71),
    ),
    axis=-1,
).reshape(-1, 3)
cp_delta_poses = jax.vmap(b3d.contact_parameters_to_pose)(delta_cps) 


test_poses = trace["object_pose_0"] @ cp_delta_poses
test_poses_batches = test_poses.split(200)

scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(
    trace, key, genjax.Pytree.const(["object_pose_0"]),
    poses) for poses in test_poses_batches
])
print(scores.max())
trace = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_pose_0"]),  test_poses[scores.argmax()])
print(trace.get_score())
b3d.rerun_visualize_trace_t(trace, 0)

# Score : 131.11410522460938
 # Inliers : 210
 # Valid : 245