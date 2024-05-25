import os
import time

import jax
import jax.numpy as jnp
from IPython import embed
from scipy.spatial.transform import Rotation as R

import b3d
import trimesh


height=100
width=100
fx=200.0
fy=200.0
cx=50.0
cy=50.0
near=0.001
far=6.0

num_frames = 60

renderer = b3d.Renderer(height, width, fx, fy, cx, cy, near, far)
poses = [b3d.Pose.from_translation(jnp.array([-3.0, 0.0, 3.5]))]
delta_pose = b3d.Pose(
    jnp.array([0.09, 0.05, 0.02]),
    R.from_euler("zyx", [-1.0, 0.1, 2.0], degrees=True).as_quat(),
)
for t in range(num_frames - 1):
    poses.append(poses[-1] @ (delta_pose))
poses = b3d.Pose.stack_poses(poses)
print("Number of frames: ", poses.shape[0])


import os
mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(trimesh.load(mesh_path))

observed_images,_ = renderer.render_attribute_many(poses[:,None], object_library.vertices, object_library.faces, jnp.array([[0, len(object_library.faces)]]), object_library.attributes)
print("observed_images.shape", observed_images.shape)

from b3d import Pose
# Defines the enumeration schedule.
key = jax.random.PRNGKey(0)
# Gridding on translation only.
translation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(lambda p: Pose.from_translation(p))(
            jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-0.01, 0.01, 5),
                    jnp.linspace(-0.01, 0.01, 5),
                    jnp.linspace(-0.01, 0.01, 5),
                ),
                axis=-1,
            ).reshape(-1, 3)
        ),
        Pose.identity()[None, ...],
    ]
)
# Sample orientations from a VMF to define a "grid" over orientations.
rotation_deltas = Pose.concatenate_poses(
    [
        jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.PRNGKey(0), 100),
            Pose.identity(),
            0.00001,
            1000.0,
        ),
        Pose.identity()[None, ...],
    ]
)


likelihood = jax.vmap(
    lambda x,y: jnp.mean(jnp.abs(x-y)), in_axes=(None, 0)
)


def update_pose_estimate(pose_estimate, gt_image):
    proposals = pose_estimate @ translation_deltas
    rendered_images,_ = renderer.render_attribute_many(proposals[:,None], object_library.vertices, object_library.faces, jnp.array([[0, len(object_library.faces)]]), object_library.attributes)
    weights_new = likelihood(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = pose_estimate @ rotation_deltas
    rendered_images,_ = renderer.render_attribute_many(proposals[:,None], object_library.vertices, object_library.faces, jnp.array([[0, len(object_library.faces)]]), object_library.attributes)
    weights_new = likelihood(gt_image, rendered_images)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

pose_estimate = poses[0]
inference_program = jax.jit(lambda p, x: jax.lax.scan(update_pose_estimate, p, x)[1])
inferred_poses = inference_program(poses[0], observed_images)

start = time.time()
pose_estimates_over_time = inference_program(poses[0], observed_images)
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", poses.shape[0] / (end - start))
 
# rerendered_images = b.RENDERER.render_many(
#     pose_estimates_over_time[:, None, ...], jnp.array([0])
# )

# viz_images = [
#     b.viz.multi_panel(
#         [
#             b.viz.scale_image(b.viz.get_depth_image(d[:, :, 2]), 3),
#             b.viz.scale_image(b.viz.get_depth_image(r[:, :, 2]), 3),
#         ],
#         labels=["Observed", "Rerendered"],
#         label_fontsize=20,
#     )
#     for (r, d) in zip(rerendered_images, observed_images)
# ]
# b.make_gif_from_pil_images(viz_images, "assets/demo.gif")

