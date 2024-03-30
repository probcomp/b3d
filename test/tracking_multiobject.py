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



import torch
from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

T = 0
output_images = interface([b3d.get_rgb_pil_image(rgbs_resized[T])])
mask  = jnp.array([jnp.array(output_image)[..., -1] > 0.5 for output_image in output_images])[0]

rr.log("/img", rr.Image(rgbs_resized[T]))
rr.log("/img/mask", rr.Image(jnp.tile((mask * 1.0)[...,None],(1,1,3))),timeless=True)
rr.log("/img", rr.Image(rgbs_resized[T] * mask[...,None]))


renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)


from scipy.ndimage.measurements import label
a, num = label(mask)

import sklearn.cluster

def segment_point_cloud(point_cloud, threshold=0.01, min_points_in_cluster=0):
    c = sklearn.cluster.DBSCAN(eps=threshold).fit(point_cloud)
    labels = c.labels_
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    counter = 0
    new_labels = np.array(labels)
    for index in order:
        if unique[index] == -1:
            continue
        if counts[index] >= min_points_in_cluster:
            val = counter
        else:
            val = -1
        new_labels[labels == unique[index]] = val
        counter += 1
    return new_labels


object_library = b3d.model.MeshLibrary.make_empty_library()

poses = []
for i in range(1,num+1):
    mask_sub = (a == i)
    xyz = video_input.xyz[0]
    point_cloud = xyz[mask_sub]
    colors = rgbs_resized[0][mask_sub]

    labels = segment_point_cloud(point_cloud, threshold=0.01, min_points_in_cluster=100)
    point_cloud = point_cloud[labels == 0]
    colors = colors[labels == 0]
    vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud, colors, point_cloud[:,2] / fx 
    )
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    segment_point_cloud
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



model = model_multiobject_gl_factory(renderer)



enumerator_observations = b3d.make_enumerator(["observed_rgb", "observed_depth"])


key = jax.random.PRNGKey(0)
translation_deltas = jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-0.01, 0.01, 11),
        jnp.linspace(-0.01, 0.01, 11),
    ),
    axis=-1,
).reshape(-1, 3))

rotation_deltas = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 400),
    Pose.identity(),
    0.00001, 1000.0
)

from dataclasses import dataclass
from typing import Any

enumerator_camera_pose = b3d.make_enumerator(["camera_pose"])
@jax.jit
def enumerative_proposal_camera(trace, key):
    key = jax.random.split(key)[0]

    test_poses = trace["camera_pose"] @ translation_deltas
    potential_scores = enumerator_camera_pose.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_camera_pose.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )

    test_poses = trace["camera_pose"] @ rotation_deltas
    potential_scores = enumerator_camera_pose.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_camera_pose.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )
    return trace, key


enumerator_0 = b3d.make_enumerator([f"object_pose_0"])
@jax.jit
def enumerative_proposal_0(trace, key):
    addr = f"object_pose_0"
    key = jax.random.split(key)[0]

    test_poses = trace[addr] @ translation_deltas
    potential_scores = enumerator_0.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_0.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )

    test_poses = trace[addr] @ rotation_deltas
    potential_scores = enumerator_0.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_0.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )
    return trace, key

enumerator_1 = b3d.make_enumerator([f"object_pose_1"])
@jax.jit
def enumerative_proposal_1(trace, key):
    addr = f"object_pose_1"
    key = jax.random.split(key)[0]

    test_poses = trace[addr] @ translation_deltas
    potential_scores = enumerator_1.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_1.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )

    test_poses = trace[addr] @ rotation_deltas
    potential_scores = enumerator_1.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_1.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )
    return trace, key

enumerator_2 = b3d.make_enumerator([f"object_pose_2"])
@jax.jit
def enumerative_proposal_2(trace, key):
    addr = f"object_pose_2"
    key = jax.random.split(key)[0]

    test_poses = trace[addr] @ translation_deltas
    potential_scores = enumerator_2.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_2.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )

    test_poses = trace[addr] @ rotation_deltas
    potential_scores = enumerator_2.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator_2.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )
    return trace, key



color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (3000.0, 3000.0)
arguments = (
    jnp.arange(3),
    color_error,
    depth_error,

    inlier_score,
    outlier_prob,

    color_multiplier,
    depth_multiplier, 
    object_library
)

START_T = 0

trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": all_poses[0],
            "object_pose_1": all_poses[1],
            "object_pose_2": all_poses[2],
            "object_0": 0,
            "object_1": 1,
            "object_2": 2,
            "observed_rgb": rgbs_resized[START_T],
            "observed_depth": xyzs[START_T,...,2],
        }
    ),
    arguments
)
b3d.rerun_visualize_trace_t(trace, 0)

END_T = len(xyzs)
key = jax.random.PRNGKey(0)
chain2 = []
for T_observed_image in tqdm(range(START_T,END_T, 1)):
    trace = enumerator_observations.update_choices(trace, key,
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    trace,key = enumerative_proposal_camera(trace, key)
    trace,key = enumerative_proposal_0(trace, key)
    trace,key = enumerative_proposal_1(trace, key)
    trace,key = enumerative_proposal_2(trace, key)
    b3d.rerun_visualize_trace_t(trace, T_observed_image)


def get_poses_from_trace(trace):
    return Pose.stack_poses([
        trace[f"object_pose_{i}"] for i in range(len(trace.get_args()[0]))
    ])

def get_object_ids_from_trace(trace):
    return jnp.array([
        trace[f"object_{i}"] for i in range(len(trace.get_args()[0]))
    ])

t  = 9
rr.set_time_sequence("frame", t)
poses = get_poses_from_trace(trace)
object_ids = get_object_ids_from_trace(trace)

jnp.conc

