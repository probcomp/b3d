import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm import tqdm
import trimesh
import pytest


PORT = 8812
rr.init("233")
rr.connect(addr=f"127.0.0.1:{PORT}")

width = 200
height = 200
fx = 200.0
fy = 200.0
cx = 100.0
cy = 100.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far, 1024)


image_width, image_height, fx, fy, cx, cy, near, far = (
    100,
    100,
    200.0,
    200.0,
    50.0,
    50.0,
    0.01,
    10.0,
)
renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
print("Vertices dimensions :", vertices.max(0) - vertices.min(0))

key = jax.random.PRNGKey(0)

camera_pose = Pose.from_position_and_target(
    jnp.array([0.6, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0])
)

cp_to_pose = lambda cp: Pose(
    jnp.array([cp[0], cp[1], 0.0]),
    b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat(),
)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)

color_error, depth_error = (60.0, 0.01)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (10000.0, 500.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)

cps_to_test = [
    jnp.array([0.0, 0.0, jnp.pi]),  # Hidden
    jnp.array([0.0, 0.0, -jnp.pi / 2]),  # Side
    jnp.array([0.0, 0.0, 0.0]),  # Front
    jnp.array([0.0, 0.0, +jnp.pi / 2]),  # Side
]

sampled_degree_range_bounds = [
    (50.0, 80.0),
    (0.0, 15.0),
    (0.0, 15.0),
    (0.0, 15.0),
]


class RGBDSensorRayModel(genjax.ExactDensity, genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, model_args, fx, fy):
        return (rendered_rgb, rendered_depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, model_args, fx, fy):
        observed_rgb, observed_depth = observed

        observed_lab = b3d.rgb_to_lab(observed_rgb)
        rendered_lab = b3d.rgb_to_lab(rendered_rgb)
        error = jnp.linalg.norm(
            observed_lab[..., 1:3] - rendered_lab[..., 1:3], axis=-1
        ) + jnp.abs(observed_lab[..., 0] - rendered_lab[..., 0])

        valid_data_mask = rendered_rgb.sum(-1) != 0.0

        color_inliers = (error < model_args.color_tolerance) * valid_data_mask
        depth_inliers = (
            jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance
        ) * valid_data_mask
        inliers = color_inliers * depth_inliers
        outliers = jnp.logical_not(inliers) * valid_data_mask
        undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)

        inlier_score = model_args.inlier_score
        outlier_prob = model_args.outlier_prob
        multiplier = model_args.color_multiplier

        corrected_depth = rendered_depth + (rendered_depth == 0.0) * 1.0
        areas = (corrected_depth / fx) * (corrected_depth / fy)

        return (
            5 * jnp.sum(inliers * areas)
            + 1.0 * jnp.sum(undecided * areas)
            + 0.01 * jnp.sum(outliers * areas)
        ) * multiplier


rgbd_sensor_ray_model = RGBDSensorRayModel()

model = b3d.model_multiobject_gl_factory(renderer, rgbd_sensor_ray_model)
importance_jit = jax.jit(model.importance)


text_index = 3
gt_cp = cps_to_test[text_index]

object_pose = cp_to_pose(gt_cp)

gt_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": camera_pose,
            "object_pose_0": object_pose,
            "object_0": 0,
            # "observed_rgb": gt_img,
            # "observed_depth": gt_depth,
        }
    ),
    (jnp.arange(1), model_args, object_library),
)
print("IMG Size :", gt_trace["observed_rgb_depth"][0].shape)

delta_cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.02, 0.02, 31),
        jnp.linspace(-0.02, 0.02, 31),
        jnp.linspace(-jnp.pi, jnp.pi, 71),
    ),
    axis=-1,
).reshape(-1, 3)
cp_delta_poses = jax.vmap(cp_to_pose)(delta_cps)

test_poses = gt_trace["object_pose_0"] @ cp_delta_poses
test_poses_batches = test_poses.split(10)
scores = jnp.concatenate(
    [
        b3d.enumerate_choices_get_scores_jit(
            gt_trace, key, genjax.Pytree.const(["object_pose_0"]), poses
        )
        for poses in test_poses_batches
    ]
)

samples = jax.random.categorical(key, scores, shape=(100,))
print("GT Contact Parameter :", gt_cp)

samples_deg_range = jnp.rad2deg(
    (jnp.max(delta_cps[samples], axis=0) - jnp.min(delta_cps[samples], axis=0))[2]
)

print("Sampled Angle Range:", samples_deg_range)


alternate_camera_pose = Pose.from_position_and_target(
    jnp.array([0.01, 0.000, 0.9]), object_pose.pos
)
alternate_view_images, _ = renderer.render_attribute_many(
    (alternate_camera_pose.inv() @ test_poses[samples])[:, None, ...],
    object_library.vertices,
    object_library.faces,
    object_library.ranges[jnp.array([0])],
    object_library.attributes,
)

for t in range(len(samples)):
    trace_ = b3d.update_choices_jit(
        gt_trace, key, genjax.Pytree.const(["object_pose_0"]), test_poses[samples[t]]
    )
    b3d.rerun_visualize_trace_t(trace_, t)
    rr.set_time_sequence("frame", t)
    rr.log("alternate_view_image", rr.Image(alternate_view_images[t, ...]))
    rr.log("text", rr.TextDocument(f"{delta_cps[samples[t]]} \n {scores[samples[t]]}"))
