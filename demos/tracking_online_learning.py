import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm

PORT = 8812
rr.init("online_learning")
rr.connect(addr=f'127.0.0.1:{PORT}')


path = os.path.join(b3d.get_assets_path(),
#  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
# "shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
"shared_data_bucket/input_data/desk_ramen2_spray1.r3d.video_input.npz")
video_input = b3d.VideoInput.load(path)

image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

rgbs = video_input.rgb[::3] / 255.0
xyzs = video_input.xyz[::3]
# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
), 0.0, 1.0)

object_library = b3d.MeshLibrary.make_empty_library()
point_cloud = xyzs[0].reshape(-1,3)
colors = rgbs_resized[0].reshape(-1,3)
sub = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(point_cloud)), (len(point_cloud),), replace=False)
point_cloud = point_cloud[sub]
colors = colors[sub]

vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud, colors, point_cloud[:,2] / fx * 4.0
)
# object_pose = Pose.from_translation(vertices.mean(0))
# vertices = object_pose.inverse().apply(vertices)
object_library.add_object(vertices, faces, vertex_colors)

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer)

color_error, depth_error = (jnp.float32(30.0), jnp.float32(0.02))
inlier_score, outlier_prob = (jnp.float32(5.0), jnp.float32(0.001))
color_multiplier, depth_multiplier = (jnp.float32(3000.0), jnp.float32(3000.0))


key = jax.random.PRNGKey(0)
translation_deltas = Pose.concatenate_poses([jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-0.01, 0.01, 11),
    ),
    axis=-1,
).reshape(-1, 3)), Pose.identity()[None,...]])

rotation_deltas = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 11*11*11),
    Pose.identity(),
    0.00001, 1000.0
), Pose.identity()[None,...]])


# for t in range(len(rgbs_resized)):
#     rr.set_time_sequence("frame", t)
#     rr.log("/rgb", rr.Image(rgbs_resized[t]))


all_deltas =  Pose.stack_poses([translation_deltas, rotation_deltas])

from functools import partial
@partial(jax.jit, static_argnames=['addressses'])
def enumerative_proposal(trace, addressses, key, all_deltas):
    addr = addressses.const[0]
    current_pose = trace[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, jax.random.PRNGKey(0), addressses, test_poses
        )
        current_pose = test_poses[potential_scores.argmax()]
    trace = b3d.update_choices(
        trace, key, addressses, current_pose
    )
    return trace, key


REAQUISITION_TS = [0, 95,222,355, len(rgbs_resized)]

importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)

START_T = 0
trace, _ = importance_jit(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict([
            ("camera_pose", Pose.identity()),
            ("object_pose_0", Pose.identity()),
            ("object_0", 0),
            ("object_1", -1),
            ("object_2", -1),
            ("object_3", -1),
            ("observed_rgb", rgbs_resized[START_T]),
            ("observed_depth", xyzs[START_T,...,2]),
        ])
    ),
    (jnp.arange(4),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library)
)
b3d.rerun_visualize_trace_t(trace, 0)
key = jax.random.PRNGKey(0)

inference_data_over_time = []
for reaquisition_phase in range(len(REAQUISITION_TS)-1):
    for T_observed_image in tqdm(range(REAQUISITION_TS[reaquisition_phase], REAQUISITION_TS[reaquisition_phase+1])):
        trace = b3d.update_choices_jit(trace, key,
            genjax.Pytree.const(["observed_rgb", "observed_depth"]),
            rgbs_resized[T_observed_image],
            xyzs[T_observed_image,...,2]
        )
        trace,key = enumerative_proposal(trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
        for i in range(1, len(object_library.ranges)):
            trace,key = enumerative_proposal(trace, genjax.Pytree.const([f"object_pose_{i}"]), key, all_deltas)
        b3d.rerun_visualize_trace_t(trace, T_observed_image)
        inference_data_over_time.append((b3d.get_poses_from_trace(trace),b3d.get_object_ids_from_trace(trace), trace["camera_pose"], T_observed_image ))

    rgb_inliers, rgb_outliers = b3d.get_rgb_inlier_outlier_from_trace(trace)
    depth_inliers, depth_outliers = b3d.get_depth_inlier_outlier_from_trace(trace)
    rr.set_time_sequence("frame", T_observed_image)
    rr.log("/rgb/rgb_outliers", rr.Image(jnp.tile((rgb_outliers*1.0)[...,None], (1,1,3))))
    rr.log("/rgb/depth_outliers", rr.Image(jnp.tile((depth_outliers*1.0)[...,None], (1,1,3))))

    outler_mask = jnp.logical_and(rgb_outliers , depth_outliers)
    rr.log("outliers", rr.Image(jnp.tile((outler_mask*1.0)[...,None], (1,1,3))))

    point_cloud = b3d.xyz_from_depth(trace["observed_depth"], fx,fy,cx,cy)[outler_mask]
    point_cloud_colors = trace["observed_rgb"][outler_mask]

    assignment = b3d.segment_point_cloud(point_cloud)

    vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud.reshape(-1,3)[assignment==0], point_cloud_colors.reshape(-1,3)[assignment==0], point_cloud.reshape(-1,3)[assignment==0][:,2] / fx * 3.0
    )

    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_library.add_object(vertices, faces, vertex_colors)

    REAQUISITION_T = REAQUISITION_TS[reaquisition_phase+1]-1
    next_object_id = len(object_library.ranges)-1
    trace = trace.update(
        key,
        genjax.choice_map({f"object_{next_object_id}": next_object_id, f"object_pose_{next_object_id}": trace["camera_pose"] @ object_pose, "observed_rgb": rgbs_resized[REAQUISITION_T], "observed_depth": xyzs[REAQUISITION_T,...,2] }),
        # genjax.Diff.tree_diff_unknown_change((jnp.arange(2), *trace.get_args()[1:]))
        genjax.Diff.tree_diff_unknown_change((jnp.arange(4),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library))
    )[0]
    b3d.rerun_visualize_trace_t(trace, REAQUISITION_T)
    inference_data_over_time.append((b3d.get_poses_from_trace(trace),b3d.get_object_ids_from_trace(trace), trace["camera_pose"], T_observed_image ))

for i in tqdm(range(len(inference_data_over_time))):
    print(t)
    poses, object_ids, camera_pose, t = inference_data_over_time[i]
    trace = update_jit(
        key,
        trace,
        genjax.choice_map(
            dict([
                *[(f"object_pose_{i}", poses[i]) for i in range(len(poses))],
                *[(f"object_{i}", object_ids[i])for i in range(len(object_ids))],
                ("camera_pose", camera_pose),
                ("observed_rgb", rgbs_resized[t]),
                ("observed_depth", xyzs[t,...,2]),
            ])
        ),
        genjax.Diff.tree_diff_unknown_change((jnp.arange(4),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library))
    )[0]
    b3d.rerun_visualize_trace_t(trace, t)
    rr.set_time_sequence("frame", t)
    outler_mask = jnp.logical_and(rgb_outliers , depth_outliers)

    rgb_inliers, rgb_outliers = b3d.get_rgb_inlier_outlier_from_trace(trace)
    depth_inliers, depth_outliers = b3d.get_depth_inlier_outlier_from_trace(trace)

    rr.log("outliers", rr.Image(jnp.tile((outler_mask*1.0)[...,None], (1,1,3))))
