import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import jax_gl_renderer
from jax.scipy.spatial.transform import Rotation as Rot
from jax_gl_renderer import Pose
import rerun as rr

rr.init("demo2.py")
rr.connect("127.0.0.1:8812")

width=100
height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = jax_gl_renderer.JaxGLRenderer(
    width, height, fx, fy, cx, cy, near, far
)

## Render color
from pathlib import Path
mesh_path = Path(jax_gl_renderer.__file__).parents[1] / "assets/006_mustard_bottle/textured_simple.obj"
mesh = trimesh.load(mesh_path)

vertices = jnp.array(mesh.vertices) * 20.0
vertices = vertices - vertices.mean(0)
faces = jnp.array(mesh.faces)
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0
ranges = jnp.array([[0, len(faces)]])

num_frames = 60

poses = [
    Pose.sample_gaussian_vmf_pose(
        jax.random.PRNGKey(15),
        Pose.from_translation(jnp.array([-2.0, 0.3, 3.5])),
        0.01, 10.0
    )
]
delta_pose = Pose(
    jnp.array([0.09, 0.05, 0.02]),
    Rot.from_euler("zyx", [-1.0, 0.1, 2.0], degrees=True).as_quat()
)
for t in range(num_frames - 1):
    poses.append(poses[-1] @ delta_pose)


all_gt_poses = Pose.stack_poses(poses)
print("Number of frames: ", all_gt_poses.shape)


_,_,_,observed_images = renderer.render_many(
    all_gt_poses.as_matrix()[:,None,...], vertices, faces, ranges)
for t in range(num_frames):
    rr.set_time_sequence("frame", t)
    rr.log("observed_image", rr.DepthImage((observed_images[t,...])))

translation_deltas = jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.2, 0.2, 5),
        jnp.linspace(-0.2, 0.2, 5),
        jnp.linspace(-0.2, 0.2, 5),
    ),
    axis=-1,
).reshape(-1, 3))

rotation_deltas = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 200),
    Pose.identity(),
    0.001, 100.0
)

def likelihood_fn(observed_depth, rendered_depth):
    return (jnp.abs(observed_depth - rendered_depth) < 0.1).sum()
likelihood_fn_parallel = jax.vmap(likelihood_fn, in_axes=(None,0))

def update_pose_estimate(pose_estimate, gt_depth):
    proposals = pose_estimate @ translation_deltas
    rendered_depth = renderer.render_many(
        proposals.as_matrix()[:,None,...], vertices, faces, ranges
    )[3]
    weights_new = likelihood_fn_parallel(gt_depth, rendered_depth)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = pose_estimate @ rotation_deltas
    rendered_depth = renderer.render_many(
        proposals.as_matrix()[:,None,...], vertices, faces, ranges
    )[3]
    weights_new = likelihood_fn_parallel(gt_depth, rendered_depth)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = jax.jit(lambda p, x: jax.lax.scan(update_pose_estimate, p, x)[1])
inferred_poses = inference_program(all_gt_poses[0], observed_images)

import time

start = time.time()
pose_estimates_over_time = inference_program(poses[0], observed_images)
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", all_gt_poses.shape[0] / (end - start))

inferred_images = renderer.render_many(pose_estimates_over_time.as_matrix()[:,None,...], vertices, faces, ranges)[3]
for t in range(num_frames):
    rr.set_time_sequence("frame", t)
    rr.log("observed_image/inferred", rr.DepthImage(inferred_images[t,...]))


pose = Pose.from_position_and_target(
    jnp.array([3.2, 0.5, 0.0]),
    jnp.array([0.0, 0.0, 0.0])

).inverse()
image = renderer.render_attribute(pose.as_matrix()[None,...], vertices, faces, ranges, vertex_colors)
rr.log("rgb_image", rr.Image(image), timeless=True)
