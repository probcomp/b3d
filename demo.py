import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import jax_gl_renderer
from jax.scipy.spatial.transform import Rotation as Rot
from jax_gl_renderer import Pose
# import rerun as rr

# rr.init("demo.py")
# rr.connect("127.0.0.1:8812")

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

def update_pose_estimate(pose_estimate, gt_image):
    proposals = pose_estimate @ translation_deltas
    rendered_images = renderer.render_depth_many(proposals.as_matrix()[:,None,...], vertices, faces, ranges)
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = pose_estimate @ rotation_deltas
    rendered_images = renderer.render_depth_many(proposals.as_matrix()[:,None,...], vertices, faces, ranges)
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = jax.jit(lambda p, x: jax.lax.scan(update_pose_estimate, p, x)[1])
inferred_poses = inference_program(all_gt_poses[0], observed_images)

import time

start = time.time()
pose_estimates_over_time_jax = inference_program(poses[0], observed_images)
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", all_gt_poses.shape[0] / (end - start))


def update_pose_estimate(pose_estimate, gt_image):
    proposals = pose_estimate @ translation_deltas
    rendered_images = renderer.render_depth_many_cuda(proposals.as_matrix()[:,None,...], vertices, faces, ranges)
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = pose_estimate @ rotation_deltas
    rendered_images = renderer.render_depth_many_cuda(proposals.as_matrix()[:,None,...], vertices, faces, ranges)
    weights_new = likelihood_fn_parallel(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = lambda p, x: jax.lax.scan(update_pose_estimate, p, x)[1]
inference_program = jax.jit(inference_program)
inferred_poses = inference_program(all_gt_poses[0], observed_images)

import time

start = time.time()
pose_estimates_over_time_cuda = inference_program(poses[0], observed_images)
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", all_gt_poses.shape[0] / (end - start))



#### hacky viz

from PIL import Image
import matplotlib.pyplot as plt
import copy
cmap  = copy.copy(plt.get_cmap('turbo'))
cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))

def get_depth_pil_image(image, min_val=None, max_val=None):
    """Convert a depth image to a PIL image.

    Args:
        image (np.ndarray): Depth image. Shape (H, W).
        min (float): Minimum depth value for colormap.
        max (float): Maximum depth value for colormap.
        cmap (matplotlib.colors.Colormap): Colormap to use.
    Returns:
        PIL.Image: Depth image visualized as a PIL image.
    """
    min_val = image.min() if min_val is None else min_val
    max_val = image.max() if max_val is None else max_val
    image = (image - min_val) / (max_val - min_val + 1e-10)

    img = Image.fromarray(
        np.rint(cmap(image) * 255.0).astype(np.int8), mode="RGBA"
    ).convert("RGB")
    return img

def make_gif_from_pil_images(images, filename):
    """Save a list of PIL images as a GIF.

    Args:
        images (list): List of PIL images.
        filename (str): Filename to save GIF to.
    """
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )

rendered_images = renderer.render_depth_many_cuda(pose_estimates_over_time_cuda.as_matrix()[:,None,...], vertices, faces, ranges)
viz_images_cuda = [get_depth_pil_image(rendered_image[:,:,2]) for rendered_image in rendered_images]
make_gif_from_pil_images(viz_images_cuda, "demo.gif")

rendered_images_jax = renderer.render_depth_many(pose_estimates_over_time_jax.as_matrix()[:,None,...], vertices, faces, ranges)
viz_images_jax = [get_depth_pil_image(rendered_image[:,:,2]) for rendered_image in rendered_images_jax]
make_gif_from_pil_images(viz_images_jax, "demo_jax.gif")




# assert correctness
assert jnp.allclose(pose_estimates_over_time_jax.as_matrix(), pose_estimates_over_time_cuda.as_matrix())


from IPython import embed; embed()