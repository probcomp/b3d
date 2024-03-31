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
rr.init("233")
rr.connect(addr=f'127.0.0.1:{PORT}')


## Render color
from pathlib import Path
import trimesh
mesh_path = Path(b3d.__file__).parents[1] / "assets/025_mug/textured_simple.obj"
mesh = trimesh.load(mesh_path)


image_width, image_height, fx, fy, cx, cy, near, far = 100, 100, 50.0, 50.0, 50.0, 50.0, 0.01, 10.0
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

vertices = jnp.array(mesh.vertices)
faces = jnp.array(mesh.faces)
vertex_colors = (jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0 ) 
# vertex_colors = vertex_colors * 0.0 + vertex_colors[0]

model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(0)

camera_pose = Pose.from_position_and_target(jnp.array([0.2, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]))

cp_to_pose = lambda cp: Pose(jnp.array([cp[0], cp[1], 0.0]), b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat())
gt_cp = jnp.array([0.0, 0.0, -jnp.pi/2])
gt_cp = jnp.array([0.0, 0.0, jnp.pi])
object_pose = cp_to_pose(gt_cp)

alternate_camera_pose = Pose.from_position_and_target(
    jnp.array([0.01, 0.000, 0.3]),
    object_pose.pos
)

object_library = b3d.model.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)

gt_img, gt_depth = renderer.render_attribute(
    (camera_pose.inv() @ object_pose).as_matrix()[None,...],
    object_library.vertices, object_library.faces,
    object_library.ranges[jnp.array([0])],
    object_library.attributes
)
rr.log("gt_img", rr.Image(gt_img), timeless=True)


color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (1000.0, 1000.0)
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



N = 1000
def accumulate(carry, x):
    (key, pose) = carry
    keys = jax.random.split(key, N)
    test_poses = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(keys, pose, 0.01, 500.0)
    scores = b3d.enumerate_choices_get_scores_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), test_poses)
    key = jax.random.split(keys[-1],2)[0]
    # pose = test_poses[jax.random.categorical(key, scores)]
    pose = test_poses[scores.argmax()]
    return (key, pose), pose
f = jax.jit(lambda arg1, arg2: jax.lax.scan(accumulate, arg1, arg2))

traces = []
for _ in range(10):
    key = jax.random.split(key, 1)[0]
    pose =  Pose.sample_gaussian_vmf_pose(key, object_pose, 0.01, 1.0)
    trace = b3d.update_choices_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), pose)
    print(trace.get_score())
    (key,pose), _ = f((key, pose), jnp.arange(50))
    trace = b3d.update_choices_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), pose)
    print(trace.get_score())
    traces.append(trace)
    b3d.rerun_visualize_trace_t(trace, 0)

trace = traces[jnp.argmax(jnp.array([t.get_score() for t in traces]))]
b3d.rerun_visualize_trace_t(trace, 0)


cp_delta_poses = jax.vmap(lambda cp: cp_to_pose(cp))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.02, 0.02, 11),
        jnp.linspace(-0.02, 0.02, 11),
        jnp.linspace(-jnp.pi/3, jnp.pi/3, 11),
    ),
    axis=-1,
).reshape(-1, 3)) 
test_poses = trace["object_pose_0"] @ cp_delta_poses

scores = b3d.enumerate_choices_get_scores_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), test_poses)
samples = jax.random.categorical(key, scores, shape=(100,))
print(samples)


alternate_view_images,_  = renderer.render_attribute_many(
    (alternate_camera_pose.inv() @  test_poses[samples[:100]]).as_matrix()[:, None,...],
    object_library.vertices, object_library.faces,
    object_library.ranges[jnp.array([0])],
    object_library.attributes
)

for t in range(100):
    trace_ = b3d.update_choices_jit(gt_trace, key,  genjax.Pytree.const(["object_pose_0"]), test_poses[samples[t]])
    b3d.rerun_visualize_trace_t(trace_, t)
    rr.set_time_sequence("frame", t)
    rr.log("alternate_view_image", rr.Image(alternate_view_images[t,...]))

