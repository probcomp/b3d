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
rr.init("asdf223ff3")
rr.connect(addr=f'127.0.0.1:{PORT}')


path = os.path.join(b3d.get_assets_path(),
#  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
"shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
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

vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud, colors, point_cloud[:,2] / fx * 4.0
)
# object_pose = Pose.from_translation(vertices.mean(0))
# vertices = object_pose.inverse().apply(vertices)
object_library.add_object(vertices, faces, vertex_colors)

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer)

color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (3000.0, 3000.0)
arguments = (jnp.arange(1),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library)


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



START_T = 0
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict([
            ("camera_pose", Pose.identity()),
            ("object_pose_0", Pose.identity()),
            ("object_0", 0),
            ("observed_rgb", rgbs_resized[START_T]),
            ("observed_depth", xyzs[START_T,...,2]),
        ])
    ),
    arguments
)
b3d.rerun_visualize_trace_t(trace, 0)
key = jax.random.PRNGKey(0)
traces = []

REAQUISITION_T = 110
for T_observed_image in tqdm(range(START_T, REAQUISITION_T)):
    trace = b3d.update_choices_jit(trace, key,
        genjax.Pytree.const(["observed_rgb", "observed_depth"]),
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    trace,key = enumerative_proposal(trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
    # for i in range(num_objects):
    #     trace,key = enumerative_proposal(trace, genjax.Pytree.const([f"object_pose_{i}"]), key, all_deltas)
    traces.append(trace)
    b3d.rerun_visualize_trace_t(trace, T_observed_image)


T = len(traces) - 1
rgb_inliers, rgb_outliers = b3d.get_rgb_inlier_outlier_from_trace(trace)
depth_inliers, depth_outliers = b3d.get_depth_inlier_outlier_from_trace(trace)
rr.set_time_sequence("frame", T)
rr.log("/rgb/rgb_outliers", rr.Image(jnp.tile((rgb_outliers*1.0)[...,None], (1,1,3))))
rr.log("/rgb/depth_outliers", rr.Image(jnp.tile((depth_outliers*1.0)[...,None], (1,1,3))))

outler_mask = jnp.logical_and(rgb_outliers , depth_outliers)
rr.log("outliers", rr.Image(jnp.tile((outler_mask*1.0)[...,None], (1,1,3))))

point_cloud = b3d.xyz_from_depth(trace["observed_depth"], fx,fy,cx,cy)[outler_mask]
point_cloud_colors = trace["observed_rgb"][outler_mask]
rr.log("pc", rr.Points3D(point_cloud.reshape(-1,3), colors=point_cloud_colors.reshape(-1,3)))

assignment = b3d.segment_point_cloud(point_cloud)
rr.log("pc", rr.Points3D(point_cloud.reshape(-1,3)[assignment==0], colors=point_cloud_colors.reshape(-1,3)[assignment==0]))

vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud.reshape(-1,3)[assignment==0], point_cloud_colors.reshape(-1,3)[assignment==0], point_cloud.reshape(-1,3)[assignment==0][:,2] / fx * 4.0
)

rr.log(
    f"/3d/mesh",
    rr.Mesh3D(
        vertex_positions=vertices,
        indices=faces,
        vertex_colors=vertex_colors
    ),
    timeless=True
)

object_pose = Pose.from_translation(vertices.mean(0))
vertices = object_pose.inverse().apply(vertices)
object_library.add_object(vertices, faces, vertex_colors)

# rr.log(
#     f"/3d/mesh",
#     rr.Mesh3D(
#         vertex_positions=vertices,
#         indices=faces,
#         vertex_colors=vertex_colors
#     ),
#     timeless=True
# )


arguments = (jnp.arange(2),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library)
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict([
            ("camera_pose", traces[-1]["camera_pose"]),
            ("object_pose_0", traces[-1]["object_pose_0"]),
            ("object_0", 0),
            ("object_pose_1", traces[-1]["camera_pose"] @ object_pose),
            ("object_1", 1),
            ("observed_rgb", rgbs_resized[REAQUISITION_T]),
            ("observed_depth", xyzs[REAQUISITION_T,...,2]),
        ])
    ),
    arguments
)
b3d.rerun_visualize_trace_t(trace, REAQUISITION_T)

traces_2 = traces
END_T = len(xyzs)
key = jax.random.PRNGKey(0)
for T_observed_image in tqdm(range(REAQUISITION_T,END_T, 1)):
    trace = b3d.update_choices_jit(trace, key,
        genjax.Pytree.const(["observed_rgb", "observed_depth"]),
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    trace,key = enumerative_proposal(trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
    trace,key = enumerative_proposal(trace, genjax.Pytree.const([f"object_pose_1"]), key, all_deltas)
    traces_2.append(trace)
    b3d.rerun_visualize_trace_t(trace, T_observed_image)

