import io
import os
import time

import b3d
import b3d.bayes3d as bayes3d
import genjax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import trimesh
from genjax import Pytree
from genjax._src.core.serialization.msgpack import msgpack_serialize
from PIL import Image


def scale_mesh(vertices, scale_factor):
    vertices[:, 0] *= scale_factor[0]
    vertices[:, 1] *= scale_factor[1]
    vertices[:, 2] *= scale_factor[2]
    return vertices


def euler_angles_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to a quaternion.

    Source: https://pastebin.com/riRLRvch

    :param euler: The Euler angles vector.

    :return: The quaternion representation of the Euler angles.
    """
    pitch = np.radians(euler[0] * 0.5)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    yaw = np.radians(euler[1] * 0.5)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    roll = np.radians(euler[2] * 0.5)
    cr = np.cos(roll)
    sr = np.sin(roll)

    x = sy * cp * sr + cy * sp * cr
    y = sy * cp * cr - cy * sp * sr
    z = cy * cp * sr - sy * sp * cr
    w = cy * cp * cr + sy * sp * sr
    return np.array([x, y, z, w])


# paths for reading physion metadata
physion_assets_path = os.path.join(
    b3d.get_root_path(),
    "assets/physion/",
)

stim_name = (
    "lf_0/dominoes_all_movies/pilot_dominoes_0mid_d3chairs_o1plants_tdwroom_0012"
)

hdf5_file_path = os.path.join(
    physion_assets_path,
    f"{stim_name}.hdf5",
)

mesh_file_path = os.path.join(
    physion_assets_path,
    "all_flex_meshes/",
)

vfov = 54.43222
near_plane = 0.1
far_plane = 100
depth_arr = []
image_arr = []
with h5py.File(hdf5_file_path, "r") as f:
    # extract depth info
    for key in f["frames"].keys():
        depth = jnp.array(f["frames"][key]["images"]["_depth_cam0"])
        depth_arr.append(depth)
        image = jnp.array(
            Image.open(io.BytesIO(f["frames"][key]["images"]["_img_cam0"][:]))
        )
        image_arr.append(image)
    depth_arr = jnp.asarray(depth_arr)
    image_arr = jnp.asarray(image_arr) / 255
    FINAL_T, height, width = image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]

    # extract camera info
    camera_azimuth = np.array(f["azimuth"]["cam_0"])
    camera_matrix = np.array(
        f["frames"]["0000"]["camera_matrices"]["camera_matrix_cam0"]
    ).reshape((4, 4))
    projection_matrix = np.array(
        f["frames"]["0010"]["camera_matrices"]["projection_matrix_cam0"]
    ).reshape((4, 4))

    # Calculate the intrinsic matrix from vertical_fov.
    # Motice that hfov and vfov are different if height != width
    # We can also get the intrinsic matrix from opengl's perspective matrix.
    # http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
    vfov = vfov / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov * width / float(height)
    fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = height / 2.0 / tan_half_vfov

    # extract object info
    object_ids = np.array(f["static"]["object_ids"])
    model_names = np.array(f["static"]["model_names"])
    assert len(object_ids) == len(model_names)
    distractors = (
        np.array(f["static"]["distractors"])
        if np.array(f["static"]["distractors"]).size != 0
        else None
    )
    occluders = (
        np.array(f["static"]["occluders"])
        if np.array(f["static"]["occluders"]).size != 0
        else None
    )
    initial_position = np.array(f["static"]["initial_position"])
    initial_rotation = np.array(f["static"]["initial_rotation"])
    scales = np.array(f["static"]["scale"])

excluded_model_ids = np.concatenate(
    (np.where(model_names == distractors), np.where(model_names == occluders)), axis=0
)
included_model_ids = [
    idx for idx in range(len(object_ids)) if idx not in excluded_model_ids
]
included_model_names = [model_names[idx] for idx in included_model_ids]

object_initial_positions = [
    pos for idx, pos in enumerate(initial_position) if idx in included_model_ids
]
object_initial_rotations = [
    rot for idx, rot in enumerate(initial_rotation) if idx in included_model_ids
]
object_scales = [scale for idx, scale in enumerate(scales) if idx in included_model_ids]
object_meshes = []
for idx, model_name in enumerate(included_model_names):
    trim = trimesh.load(
        os.path.join(mesh_file_path, f"{model_name.decode('UTF-8')}.obj")
    )
    object_meshes.append((scale_mesh(trim.vertices, object_scales[idx]), trim.faces))

all_object_poses = []
for idx in range(len(included_model_ids)):
    object_pose = b3d.Pose(
        jnp.asarray(object_initial_positions[idx]),
        jnp.asarray(euler_angles_to_quaternion(object_initial_rotations[idx])),
    )
    all_object_poses.append(object_pose)

# load original meshes without scaling
object_library = bayes3d.MeshLibrary.make_empty_library()

for obj in object_meshes:
    vertex_colors = jnp.full(obj[0].shape, 0.4)
    object_library.add_object(obj[0], obj[1], vertex_colors)

R = camera_matrix[:3, :3]
T = camera_matrix[0:3, 3]
a = np.array([-R[0, :], -R[1, :], -R[2, :]])
b = np.array(T)
camera_position_from_matrix = np.linalg.solve(a, b)
camera_rotation_from_matrix = -np.transpose(R)
camera_pose = b3d.Pose(
    camera_position_from_matrix,
    b3d.Rot.from_matrix(camera_rotation_from_matrix).as_quat(),
)

# Defines the enumeration schedule.
key = jax.random.PRNGKey(0)
renderer = b3d.Renderer(
    width=width,
    height=height,
    fx=fx,
    fy=fy,
    cx=width / 2,
    cy=height / 2,
    near=near_plane,
    far=far_plane,
)
model = bayes3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (1e100, 0.1)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (10000.0, 500.0)
model_args = bayes3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)

# Initial trace for timestep 0
START_T = 0
trace, _ = importance_jit(
    jax.random.PRNGKey(0),
    genjax.ChoiceMap.d(
        dict(
            [
                ("camera_pose", camera_pose),
                ("object_pose_0", all_object_poses[0]),
                ("object_pose_1", all_object_poses[1]),
                ("object_pose_2", all_object_poses[2]),
                ("object_0", 0),
                ("object_1", 1),
                ("object_2", 2),
                (
                    "observed_rgb_depth",
                    (np.flip(image_arr[START_T], 1), np.flip(depth_arr[START_T], 1)),
                ),
            ]
        )
    ),
    (jnp.arange(3), model_args, object_library),
)
print("finished initialization!")

# Gridding on translation only.
translation_deltas = b3d.Pose.concatenate_poses(
    [
        jax.vmap(lambda p: b3d.Pose.from_translation(p))(
            jnp.stack(
                jnp.meshgrid(
                    jnp.linspace(-0.1, 0.1, 11),
                    jnp.linspace(-0.1, 0.1, 11),
                    jnp.linspace(-0.1, 0.1, 11),
                ),
                axis=-1,
            ).reshape(-1, 3)
        ),
        b3d.Pose.identity()[None, ...],
    ]
)
# Sample orientations from a VMF to define a "grid" over orientations.
rotation_deltas = b3d.Pose.concatenate_poses(
    [
        jax.vmap(b3d.Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.PRNGKey(0), 11 * 11 * 11),
            b3d.Pose.identity(),
            0.00001,
            1000.0,
        ),
        b3d.Pose.identity()[None, ...],
    ]
)
all_deltas = b3d.Pose.stack_poses([translation_deltas, rotation_deltas])

FINAL_T = len(image_arr)
for T_observed_image in range(FINAL_T):
    start = time.time()
    # Constrain on new RGB and Depth data.
    trace = b3d.update_choices(
        trace,
        Pytree.const(("observed_rgb_depth",)),
        (
            np.flip(image_arr[T_observed_image], 1),
            np.flip(depth_arr[T_observed_image], 1),
        ),
    )
    print("updated choices")
    trace, key = bayes3d.enumerate_and_select_best_move(
        trace, Pytree.const(("object_pose_0",)), key, all_deltas
    )
    print("searched object_pose_0")
    trace, key = bayes3d.enumerate_and_select_best_move(
        trace, Pytree.const(("object_pose_1",)), key, all_deltas
    )
    print("searched object_pose_1")
    trace, key = bayes3d.enumerate_and_select_best_move(
        trace, Pytree.const(("object_pose_2",)), key, all_deltas
    )
    print("searched object_pose_2")
    with open(
        f"/home/hlwang_ipe_genjax/b3d/saved_traces/{T_observed_image}.pickle", "wb"
    ) as output_file:
        msgpack_serialize.dump(trace, output_file)
    end = time.time()
    print(f"{T_observed_image}/{FINAL_T} -- {end - start}")
