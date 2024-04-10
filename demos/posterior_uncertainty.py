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
rr.init("233")
rr.connect(addr=f'127.0.0.1:{PORT}')


mesh_path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/mug.obj")
mesh = trimesh.load(mesh_path)

image_width, image_height, fx, fy, cx, cy, near, far = 100, 100, 200.0, 200.0, 50.0, 50.0, 0.01, 10.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

vertices = jnp.array(mesh.vertices) / 200.0
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])

model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(0)

camera_pose = Pose.from_position_and_target(jnp.array([0.6, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]))

cp_to_pose = lambda cp: Pose(jnp.array([cp[0], cp[1], 0.0]), b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat())
gt_cp = jnp.array([0.0, 0.0, -jnp.pi/2])
# gt_cp = jnp.array([0.0, 0.0, jnp.pi/2])
# gt_cp = jnp.array([0.0, 0.0, jnp.pi])
# gt_cp = jnp.array([0.0, 0.0, 0.0])
object_pose = cp_to_pose(gt_cp)

alternate_camera_pose = Pose.from_position_and_target(
    jnp.array([0.01, 0.000, 0.9]),
    object_pose.pos
)

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)

gt_img, gt_depth = renderer.render_attribute(
    (camera_pose.inv() @ object_pose)[None,...],
    object_library.vertices, object_library.faces,
    object_library.ranges[jnp.array([0])],
    object_library.attributes
)
rr.log("gt_img", rr.Image(gt_img), timeless=True)


rr.log(
    "/3d/mesh",
    rr.Mesh3D(
        vertex_positions=vertices,
        indices=faces,
        vertex_colors=vertex_colors,
    ),
    timeless=False,
)


rr.log(
    "/3d/vertices",
    rr.Points3D(
        vertices
    ),
    timeless=False,
)




color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (300.0, 300.0)
arguments = (
        jnp.arange(1),
        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier,
        object_library
    )


gt_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": camera_pose,
            "object_pose_0": object_pose,
            "object_0": 0,
            "observed_rgb": gt_img,
            "observed_depth": gt_depth,
        }
    ),
    arguments
)
b3d.rerun_visualize_trace_t(gt_trace, 0)

cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.03, 0.03, 41),
        jnp.linspace(-0.03, 0.03, 41),
        jnp.linspace(-jnp.pi, jnp.pi, 61),
    ),
    axis=-1,
).reshape(-1, 3)
cp_delta_poses = jax.vmap(lambda cp: cp_to_pose(cp))(cps) 

def pose_array_split(poses, n):
    return [Pose(ps,qs) for (ps,qs) in zip(jnp.array_split(poses.pos, n), jnp.array_split(poses.quat, n))]

test_poses = gt_trace["object_pose_0"] @ cp_delta_poses
test_poses_batches = pose_array_split(test_poses, 10)
scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), poses) for poses in test_poses_batches])
samples = jax.random.categorical(key, scores, shape=(100,))

samples = jnp.argsort(scores)[::-1][:500]

samples = jax.random.categorical(key, scores, shape=(500,))
viz_poses = test_poses[samples]

alternate_view_images,_  = renderer.render_attribute_many(
    (alternate_camera_pose.inv() @  test_poses[samples])[:, None,...],
    object_library.vertices, object_library.faces,
    object_library.ranges[jnp.array([0])],
    object_library.attributes
)

for t in range(len(samples)):
    trace_ = b3d.update_choices_jit(gt_trace, key,  genjax.Pytree.const(["object_pose_0"]), test_poses[samples[t]])
    b3d.rerun_visualize_trace_t(trace_, t)
    rr.set_time_sequence("frame", t)
    rr.log("alternate_view_image", rr.Image(alternate_view_images[t,...]))
    rr.log("text", rr.TextDocument(f"{cps[samples[t]]} \n {scores[samples[t]]}"))

