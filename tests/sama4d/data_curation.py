import os

import b3d
import jax
import jax.numpy as jnp
import trimesh


def get_loaders_for_all_curated_scenes():
    """
    Returns a list of dictionaries, each containing keys `scene_name` (string)
    and `feature_track_data_loader`.
    The value at key `feature_track_data_loader` is a function of 0 arguments;
    when called it returns a `b3d.io.FeatureTrackData` object.
    (A function is returned rather than the `b3d.io.FeatureTrackData` object itself
    to avoid loading the data until it is needed.)
    """
    scene_loaders = []
    scene_loaders.extend(get_loaders_for_curated_unity_scenes())
    scene_loaders.append(get_cheezitbox_scene_loader())
    return scene_loaders


### Curated Unity Scenes ###


def get_loaders_for_curated_unity_scenes():
    return [
        {
            "scene_name": spec["scene_name"],
            "feature_track_data_loader": (
                lambda: feature_track_data_from_scene_spec(spec)
            ),
        }
        for spec in get_curated_unity_scene_specifications()
    ]


def feature_track_data_from_scene_spec(spec):
    return (
        b3d.io.FeatureTrackData.load(spec["path"])
        .slice_time(start_frame=spec["start_frame"])
        .downscale(spec["downscale_factor"])
<<<<<<< HEAD
=======
        .flip_xy()
>>>>>>> main
    )


def get_curated_unity_scene_specifications():
    # These are filenames in 'shared_data_bucket/input_data/unity/keypoints/indoorplant/'
    good_filename_starttime_pairs = [
        ("plantRoomLookingThrough_30fps_lit_bg_800p.input.npz", 0),
        ("slidingBooks_60fps_lit_bg_800p.input.npz", 21),
        ("slidingPiledBooks_60fps_lit_bg_800p.input.npz", 21),
    ]
    return [
        {
            "scene_name": filename,
            "path": os.path.join(
                b3d.get_assets_path(),
                "shared_data_bucket/dynamic_SfM/feature_track_data",
                filename,
            ),
            "start_frame": starttime,
            "downscale_factor": 4,  # 800 x 800 -> 200 x 200
        }
        for filename, starttime in good_filename_starttime_pairs
    ]


### Scenes Manually Constructed in Python ###


def get_cheezitbox_scene_loader(n_frames=30):
    return {
        "scene_name": "rotating_cheezit_box",
        "feature_track_data_loader": (
            lambda: ftd_from_rotating_cheezit_box(n_frames=n_frames)
        ),
    }


# Scene manually constructed in Python: rotating cheezit box
def ftd_from_rotating_cheezit_box(n_frames=30):
    (r, centers_2D_frame_0, centers_3D_W_over_time, poses_WC, observed_rgbds) = (
        load_rotating_cheezit_box_data(n_frames)
    )
    return b3d.io.FeatureTrackData(
        observed_keypoints_positions=jax.vmap(
            lambda positions_3D_W, X_WC: b3d.xyz_to_pixel_coordinates(
                X_WC.inv().apply(positions_3D_W), r.fx, r.fy, r.cx, r.cy
            ),
            in_axes=(0, 0),
        )(centers_3D_W_over_time, poses_WC),
        keypoint_visibility=jnp.ones(
            (n_frames, centers_2D_frame_0.shape[0]), dtype=bool
        ),
        camera_intrinsics=r.get_intrinsics_object().as_array(),
        rgbd_images=observed_rgbds,
        latent_keypoint_positions=centers_3D_W_over_time,
        camera_position=poses_WC.pos,
        camera_quaternion=poses_WC.xyzw,
        # Every point is assigned to one object (the cheez-it box)
        object_assignments=jnp.zeros(centers_2D_frame_0.shape[0], dtype=int),
    )


def load_rotating_cheezit_box_data(n_frames=30):
    renderer = get_default_renderer()

    mesh_path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj",
    )
    mesh = trimesh.load(mesh_path)
    cheezit_object_library = b3d.MeshLibrary.make_empty_library()
    cheezit_object_library.add_trimesh(mesh)
    box_poses_W = vec_transform_axis_angle(
        jnp.array([0, 0, 1]), jnp.linspace(jnp.pi / 4, 3 * jnp.pi / 4, 30)
    )
    box_poses_W = b3d.Pose.from_matrix(box_poses_W)
    box_poses_W = box_poses_W[:n_frames]
    cam_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.15, 0.15, 0.0]), jnp.array([0.0, 0.0, 0.0])
    )
    X_WC = cam_pose
    box_poses_C = X_WC.inv() @ box_poses_W

    rgbs, depths = renderer.render_attribute_many(
        box_poses_C[:, None, ...],
        cheezit_object_library.vertices,
        cheezit_object_library.faces,
        jnp.array([[0, len(cheezit_object_library.faces)]]),
        cheezit_object_library.attributes,
    )
    observed_rgbds = jnp.concatenate([rgbs, depths[..., None]], axis=-1)
    xyzs_C = b3d.utils.xyz_from_depth_vectorized(
        depths, renderer.fx, renderer.fy, renderer.cx, renderer.cy
    )

    # Values are in pixel space, in order (H, W)
    # width_gradations = jnp.arange(46, 80, 8) - 12
    # height_gradations = jnp.arange(42, 88, 8) - 12
    width_gradations = jnp.arange(36, 70, 6)
    height_gradations = jnp.arange(30, 76, 6)
    centers_2D_frame_0 = all_pairs_2(height_gradations, width_gradations)

    centers_3D_frame0_C = xyzs_C[0][centers_2D_frame_0[:, 0], centers_2D_frame_0[:, 1]]
    centers_3D_frame0_W = X_WC.apply(centers_3D_frame0_C)

    # Let frame B0 be the first box pose
    X_W_B0 = box_poses_W[0]
    centers_3D_B0 = X_W_B0.inv().apply(centers_3D_frame0_W)
    centers_3D_W_over_time = jax.vmap(lambda X_W_Bt: X_W_Bt.apply(centers_3D_B0))(
        box_poses_W
    )

    poses_WC = jax.vmap(lambda x: X_WC)(jnp.arange(n_frames))

    return (
        renderer,
        centers_2D_frame_0,
        centers_3D_W_over_time,
        poses_WC,
        observed_rgbds,
    )


### Utils ###


def all_pairs_2(X, Y):
    return jnp.swapaxes(jnp.stack(jnp.meshgrid(X, Y), axis=-1), 0, 1).reshape(-1, 2)


def rotation_from_axis_angle(axis, angle):
    """Creates a rotation matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The rotation matrix. Shape (3, 3)
    """
    sina = jnp.sin(angle)
    cosa = jnp.cos(angle)
    direction = axis / jnp.linalg.norm(axis)
    # rotation matrix around unit vector
    R = jnp.diag(jnp.array([cosa, cosa, cosa]))
    R = R + jnp.outer(direction, direction) * (1.0 - cosa)
    direction = direction * sina
    R = R + jnp.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return R


def transform_from_rot(rotation):
    """Creates a pose matrix from a rotation matrix.

    Args:
        rotation (jnp.ndarray): The rotation matrix. Shape (3, 3)
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return jnp.vstack(
        [jnp.hstack([rotation, jnp.zeros((3, 1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )


def transform_from_axis_angle(axis, angle):
    """Creates a pose matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return transform_from_rot(rotation_from_axis_angle(axis, angle))


# calculate sequence of pose transformations
r_mat = transform_from_axis_angle(jnp.array([0, 0, 1]), jnp.pi / 2)
vec_transform_axis_angle = jax.vmap(transform_from_axis_angle, (None, 0))


def get_default_renderer():
    image_width = 120
    image_height = 100
    fx = 50.0
    fy = 50.0
    cx = 50.0
    cy = 50.0
    near = 0.001
    far = 16.0
    return b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
