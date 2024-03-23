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

rr.init("demo.py")
rr.connect("127.0.0.1:8812")

width=100
height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=6.0
renderer = jax_gl_renderer.JaxGLRenderer(
    width, height, fx, fy, cx, cy, near, far
)

box_mesh = trimesh.creation.box()
vertices = jnp.array(box_mesh.vertices) * 2.0
faces = jnp.array(box_mesh.faces)
ranges = jnp.array([[0, len(faces)]])

num_frames = 60

poses = [
    Pose.from_translation(jnp.array([-3.0, 0.0, 3.5]))
]
delta_pose = Pose(
    jnp.array([0.09, 0.05, 0.02]),
    Rot.from_euler("zyx", [-1.0, 0.1, 2.0], degrees=True).as_quat()
)
for t in range(num_frames - 1):
    poses.append(poses[-1] @ delta_pose)


all_gt_poses = Pose.stack_poses(poses)
print("Number of frames: ", all_gt_poses.shape)

observed_images = renderer.render_depth_many(all_gt_poses.as_matrix()[:,None,...], vertices, faces, ranges)
for t in range(num_frames):
    rr.set_time_sequence("frame", t)
    rr.log("observed_image", rr.DepthImage(observed_images[t,...,2]))

translation_deltas = jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.2, 0.2, 5),
        jnp.linspace(-0.2, 0.2, 5),
        jnp.linspace(-0.2, 0.2, 5),
    ),
    axis=-1,
).reshape(-1, 3))

rotation_deltas = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 100),
    Pose.identity(),
    0.00001, 800.0
)

def likelihood_fn(observed_depth, rendered_depth):
    return (jnp.abs(observed_depth[...,2] - rendered_depth[...,2]) < 0.02).sum()
likelihood_fn_parallel = jax.vmap(likelihood_fn, in_axes=(None,0))




def render_depth(
    poses, vertices, faces, ranges
):
    uvs,_,triangle_ids = renderer.render_to_barycentrics_many(poses.as_matrix()[:,None,...], vertices, faces, ranges)
    modified_vertices = jax.vmap(
        lambda i: poses[i].apply(vertices)
    )(jnp.arange(poses.shape[0]))
    rendered_images = renderer.interpolate(
        modified_vertices[0], uvs, triangle_ids, faces
    )
    return rendered_images

def update_pose_estimate(pose_estimate, gt_image):
    proposals = pose_estimate @ translation_deltas
    rendered_images = render_depth(
        proposals, vertices, faces, ranges
    )
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = pose_estimate @ rotation_deltas
    rendered_images = render_depth(
        proposals, vertices, faces, ranges
    )
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
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

inferred_images = renderer.render_depth_many(pose_estimates_over_time.as_matrix()[:,None,...], vertices, faces, ranges)
for t in range(num_frames):
    rr.set_time_sequence("frame", t)
    rr.log("observed_image/inferred", rr.DepthImage(inferred_images[t,...,2]))
