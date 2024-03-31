import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm
from b3d.model import model_gl_factory, model_multiobject_gl_factory

PORT = 8812
rr.init("asdf223ff3")
rr.connect(addr=f'127.0.0.1:{PORT}')


path = os.path.join(b3d.get_assets_path(),
#  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
"shared_data_bucket/input_data/ramen_ramen_mug_2.r3d.video_input.npz")
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

T = 0
foreground_mask = b3d.nn_background_segmentation([b3d.get_rgb_pil_image(rgbs_resized[T])])[0]

rr.log("/img", rr.Image(rgbs_resized[T]))
rr.log("/img/mask", rr.Image(jnp.tile((foreground_mask * 1.0)[...,None],(1,1,3))),timeless=True)


renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = model_multiobject_gl_factory(renderer)

from scipy.ndimage import label

assignments, num_objects = label(foreground_mask)
object_library = b3d.MeshLibrary.make_empty_library()

poses = []
vertices_lens = []
for i in range(num_objects):
    xyz = video_input.xyz[0]
    point_cloud = xyz[(assignments == i+1)]
    colors = rgbs_resized[0][(assignments == i+1)]

    labels = b3d.segment_point_cloud(point_cloud, threshold=0.01, min_points_in_cluster=100)
    point_cloud = point_cloud[labels == 0]
    colors = colors[labels == 0]
    sub = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(point_cloud)), (200,), replace=False)
    vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud, colors, point_cloud[:,2] / fx * 2.0
    )
    vertices_lens.append(len(vertices))
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    poses.append(object_pose)
    object_library.add_object(vertices, faces, vertex_colors)

    rr.log(
        f"/3d/mesh/{i}",
        rr.Mesh3D(
            vertex_positions=vertices,
            indices=faces,
            vertex_colors=vertex_colors
        ),
        timeless=True
    )
all_poses = Pose.stack_poses(poses)


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

color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (3000.0, 3000.0)
arguments = (jnp.arange(3),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library)

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
            *[(f"object_pose_{i}", all_poses[i]) for i in range(num_objects)],
            *[(f"object_{i}", i) for i in range(num_objects)],
            ("observed_rgb", rgbs_resized[START_T]),
            ("observed_depth", xyzs[START_T,...,2]),
        ])
    ),
    arguments
)
b3d.rerun_visualize_trace_t(trace, 0)



END_T = len(xyzs)
key = jax.random.PRNGKey(0)
traces = []
for T_observed_image in tqdm(range(START_T,END_T, 1)):
    trace = b3d.update_choices_jit(trace, key,
        genjax.Pytree.const(["observed_rgb", "observed_depth"]),
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    trace,key = enumerative_proposal(trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
    for i in range(num_objects):
        trace,key = enumerative_proposal(trace, genjax.Pytree.const([f"object_pose_{i}"]), key, all_deltas)
    traces.append(trace)
    b3d.rerun_visualize_trace_t(trace, T_observed_image)


T = 0
trace = traces[T]

for T in range(len(traces)):
    rgb_inliers, rgb_outliers = b3d.get_rgb_inlier_outlier_from_trace(traces[T])
    depth_inliers, depth_outliers = b3d.get_depth_inlier_outlier_from_trace(traces[T])
    rr.set_time_sequence("frame", T)
    rr.log("/rgb/rgb_outliers", rr.Image(jnp.tile((rgb_outliers*1.0)[...,None], (1,1,3))))
    rr.log("/rgb/depth_outliers", rr.Image(jnp.tile((depth_outliers*1.0)[...,None], (1,1,3))))


for trace in traces:
    object_poses = b3d.get_poses_from_trace(trace)
    for i in range(num_objects):
        rr.log(
            f"/3d/mesh/{i}",
            rr.Mesh3D(
                vertex_positions=object_library.vertices,
                indices=object_library.faces,
                vertex_colors=object_library.vertex_colors,
                pose=object_poses[i].as_matrix()
            ),
            timeless=True
        )





object_poses = b3d.get_poses_from_trace(trace)

center_point = object_poses.pos.mean(0) * jnp.array([0.0, 0.0, 1.0])

waypoints = [
    jnp.zeros(3),
    jnp.array([0.5, -0.7, 0.6]),
]



trajectory = jnp.concatenate([jnp.linspace(waypoints[i], waypoints[i+1], int(jnp.linalg.norm(waypoints[i] - waypoints[i+1]) / 0.025) ) for i in range(len(waypoints)-1)])


view_points = jax.vmap(lambda position: 
                       Pose.from_position_and_target(
                            position,
                            center_point,
                            jnp.array([0.0, -1.0, 0.0])
                        ).inv() )(trajectory)

camera_frame_poses = view_points[:,None,...] @ object_poses
images, _ = renderer.render_attribute_many(
    camera_frame_poses.as_matrix(),
    object_library.vertices,
    object_library.faces,
    object_library.ranges[jnp.array([0,1,2])],
    object_library.attributes,
)

for t in range(len(images)):
    rr.set_time_sequence("frame", t)
    rr.log(f"/img", rr.Image(images[t]))

