from tests.common.task import Task
import jax
import jax.numpy as jnp
import os
import b3d
import trimesh

class PatchTrackingTask(Task):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - camera_pose [known fixed camera pose]
        - initial_patch_positions_2D [2D patch center positions at frame 0]
            (N, 2) array of 2D patch center positions at frame 0

    The "ground truth" data consists of
        - patch_positions_3D [3D patch center positions at each frame]
            (T, N, 3) array

    A "solution" to the task looks like
        - inferred_patch_positions_3D [3D patch center positions at each frame]
            (T, N, 3) array

    Indexing in the `N` dimension in any of these arrays will index to the same keypoint.

    The task is scored by comparing the inferred_patch_positions_3D to the patch_positions_3D.
    """
    def __init__(self,
        video, X_WC, initial_patch_positions_2D, patch_positions_3D,
        renderer=None
    ):
        self.video = video
        self.X_WC = X_WC
        self.initial_patch_positions_2D = initial_patch_positions_2D
        self.patch_positions_3D = patch_positions_3D
        if renderer is None:
            renderer = self.get_default_renderer()
        self.renderer = renderer

    def get_task_specification(self):
        return {
            "video": self.video,
            "camera_pose": self.X_WC,
            "initial_patch_positions_2D": self.initial_patch_positions_2D
        }

    def score(self, inferred_patch_positions_3D,
        distance_error_threshold=0.1
    ):
        return {
            "mean_distance_error": jnp.mean(
                jnp.linalg.norm(inferred_patch_positions_3D - self.patch_positions_3D, axis=-1)
            ),
            "n_errors_above_threshold_per_frame": jnp.sum(
                jnp.linalg.norm(inferred_patch_positions_3D - self.patch_positions_3D, axis=-1) > distance_error_threshold,
                axis=-1
            )
        }

    def assert_passing(self, metrics):
        n_tracks = self.patch_positions_3D.shape[1]
        assert metrics["n_errors_above_threshold_per_frame"] < n_tracks * 0.1

    def visualize_task(self):
        pass

    def visualize_solution(self, solution, metrics):
        pass

    ### Helpers ###
    @classmethod
    def task_from_rotating_cheezit_box(cls):
        (renderer, centers_2D_frame_0, centers_3D_W_over_time, X_WC, observed_rgbds) = cls.load_rotating_cheezit_box_data()
        return cls(observed_rgbds, X_WC, centers_2D_frame_0, centers_3D_W_over_time, renderer=renderer)

    @classmethod
    def load_rotating_cheezit_box_data(cls):
        renderer = cls.get_default_renderer()

        mesh_path = os.path.join(b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
        mesh = trimesh.load(mesh_path)
        cheezit_object_library = b3d.MeshLibrary.make_empty_library()
        cheezit_object_library.add_trimesh(mesh)
        box_poses_W = vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
        box_poses_W = b3d.Pose.from_matrix(box_poses_W)
        cam_pose = b3d.Pose.from_position_and_target(
            jnp.array([0.15, 0.15, 0.0]),
            jnp.array([0.0, 0.0, 0.0])
        )
        X_WC = cam_pose
        box_poses_C = X_WC.inv() @ box_poses_W

        rgbs, depths = renderer.render_attribute_many(
            box_poses_C[:,None,...],
            cheezit_object_library.vertices,
            cheezit_object_library.faces,
            jnp.array([[0, len(cheezit_object_library.faces)]]),
            cheezit_object_library.attributes
        )
        observed_rgbds = jnp.concatenate([rgbs, depths[...,None]], axis=-1)
        xyzs_C = b3d.utils.xyz_from_depth_vectorized(
            depths, renderer.fx, renderer.fy, renderer.cx, renderer.cy
        )
        xyzs_W = X_WC.apply(xyzs_C)

        # Values are in pixel space
        centers_2D_frame_0 = b3d.chisight.dense.patch_tracking.get_default_patch_centers()
        
        (_, _, _, _, patch_points_C) = b3d.chisight.dense.patch_tracking.get_patches_from_pointcloud(
            centers_2D_frame_0, rgbs, xyzs_W, X_WC, renderer.fx
        )
        centers_3D_frame0_W = X_WC.apply(patch_points_C)

        # Let frame B0 be the first box pose
        X_W_B0 = box_poses_W[0]
        centers_3D_B0 = X_W_B0.inv().apply(centers_3D_frame0_W)
        centers_3D_W_over_time = jax.vmap(
            lambda X_W_Bt: X_W_Bt.inv().apply(centers_3D_B0)
        )(box_poses_W)

        return (renderer, centers_2D_frame_0, centers_3D_W_over_time, X_WC, observed_rgbds)

    @staticmethod
    def get_default_renderer():
        image_width = 120; image_height = 100; fx = 50.0; fy = 50.0
        cx = 50.0; cy = 50.0; near = 0.001; far = 16.0
        return b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)


### Utils ###

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
r_mat = transform_from_axis_angle(jnp.array([0,0,1]), jnp.pi/2)
vec_transform_axis_angle = jax.vmap(transform_from_axis_angle, (None, 0))
