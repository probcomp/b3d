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

model = b3d.model_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(0)
enumerator = b3d.make_enumerator(["object_pose"])


camera_pose = Pose.from_position_and_target(jnp.array([0.2, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]))

cp_to_pose = lambda cp: Pose(jnp.array([cp[0], cp[1], 0.0]), b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat())
gt_cp = jnp.array([0.0, 0.0, jnp.pi])
gt_cp = jnp.array([0.0, 0.0, 0])
object_pose = cp_to_pose(gt_cp)
gt_img, _= renderer.render_attribute(
    (camera_pose.inv() @ object_pose).as_matrix()[None,...],
    vertices, faces,
    jnp.array([[0, len(faces)]]),
    vertex_colors
)
rr.log("gt_img", rr.Image(gt_img), timeless=True)


alternate_camera_pose = Pose.from_position_and_target(
    jnp.array([0.01, 0.000, 0.3]),
    object_pose.pos
)

color_error, depth_error = (40.0, 0.01)
inlier_score, outlier_prob = (2.0, 0.001)
color_multiplier, depth_multiplier = (2000.0, 0.0)
arguments = (
        vertices, faces, vertex_colors,
        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier
    )


gt_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": camera_pose,
            "object_pose": object_pose,
            "observed_rgb": gt_img,
        }
    ),
    arguments
)

b3d.rerun_visualize_trace_t(gt_trace, 0)

test_poses = jax.vmap(lambda cp: cp_to_pose(cp))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-jnp.pi/5, jnp.pi/5, 11),
    ),
    axis=-1,
).reshape(-1, 3) + gt_cp)

scores = enumerator.enumerate_choices_get_scores(gt_trace, key, test_poses)
samples = jax.random.categorical(key, scores, shape=(100,))
print(samples)
for i in range(100):
    trace = enumerator.update_choices(gt_trace, key, test_poses[samples[i]])
    b3d.rerun_visualize_trace_t(trace, i)

alternate_view_images,_ = renderer.render_attribute_many(
    (alternate_camera_pose.inv() @ test_poses[samples[:100]]).as_matrix()[:,None,...],
    vertices, faces,
    jnp.array([[0, len(faces)]]),
    vertex_colors
)
for t in range(100):
    rr.set_time_sequence("frame", t)
    rr.log("alternate_view_image", rr.Image(alternate_view_images[t,...]))


var,conc = 0.005, 30.0


def accumulate(key, x):
    keys = jax.random.split(key,2000)
    test_poses = jax.vmap(b3d.gaussian_vmf_pose.sample,
        in_axes=(0,None, None, None))(
        keys,
        gt_trace["object_pose"],
        var,conc
    )
    scores = enumerator.enumerate_choices_get_scores(gt_trace, keys[-1], test_poses)
    key = jax.random.split(keys[-1], 1)[0]
    return key, (test_poses, scores)

_,(poses, scores) =jax.lax.scan(accumulate, key, jnp.arange(100))
poses = Pose(poses.pos.reshape(-1,3), poses.quat.reshape(-1,4))
scores = scores.reshape(-1)


order = jnp.argsort(scores)[::-1]

for i in range(100):
    trace = enumerator.update_choices(gt_trace, key, poses[order[i]])
    b3d.rerun_visualize_trace_t(trace, i)


samples = jax.random.categorical(key, jnp.log(scores), shape=(100,))
print(samples)
order = jnp.argsort(scores)[::-1]

for i in range(100):
    trace = enumerator.update_choices(gt_trace, key, test_poses[order[i]])
    b3d.rerun_visualize_trace_t(trace, i)



for i in range(100):
    trace = enumerator.update_choices(gt_trace, key, test_poses[i])
    b3d.rerun_visualize_trace_t(trace, i)


