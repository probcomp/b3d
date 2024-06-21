from tests.common.task import Task
import jax
import jax.numpy as jnp
import os
import b3d
import trimesh
import rerun as rr

class PatchTrackingTask(Task):
    """
    The task specification consists of:
        - video [RGB or RGBD video]
        - camera_pose [known fixed camera pose]
        - initial_patch_positions_2D [2D patch center positions at frame 0]
            (N, 2) array of 2D patch center positions at frame 0
            stored as (y, x) pixel coordinates
        - renderer [Renderer object]

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
            "initial_patch_positions_2D": self.initial_patch_positions_2D,
            "renderer": self.renderer
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

    def assert_passing(self, metrics, **kwargs):
        n_tracks = self.patch_positions_3D.shape[1]
        assert jnp.all(metrics["n_errors_above_threshold_per_frame"] < n_tracks * 0.1)
        print("test ran")

    def visualize_task(self):
        rr.log("/task/frame0", rr.Image(self.video[0, :, :, :3]), timeless=True)
        rr.log("/task/initial_patch_positions_2D",
               rr.Points2D(self.initial_patch_positions_2D[:, ::-1], colors=jnp.array([0., 1., 0.])), timeless=True
            )

        for i in range(self.patch_positions_3D.shape[0]):
            rr.set_time_sequence("frame", i)
            rr.log("/task/patch_positions_3D", rr.Points3D(
                self.patch_positions_3D[i], colors=jnp.array([0., 1., 0.]), radii = 0.003)
            )
            rr.log("/task/video/rgb", rr.Image(self.video[i, :, :, :3]))
        
            if self.video.shape[-1] == 4:
                rr.log("/task/video/depth", rr.DepthImage(self.video[i, :, :, 3]))
                # If video is RGBD, get the point cloud and visualize it in the 3D viewer
                r = self.renderer
                xyzs_C = b3d.utils.xyz_from_depth_vectorized(
                    self.video[i, :, :, 3], r.fx, r.fy, r.cx, r.cy
                )
                rgbs = self.video[i, :, :, :3]
                xyzs_W = self.X_WC.apply(xyzs_C)
                rr.log("/task/observed_pointcloud", rr.Points3D(
                    positions=xyzs_W.reshape(-1,3),
                    colors=rgbs.reshape(-1,3),
                    radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
                )

    def visualize_solution(self, solution, metrics):
        pass

    ### Helpers ###
    @staticmethod
    def get_default_renderer():
        image_width = 120; image_height = 100; fx = 50.0; fy = 50.0
        cx = 50.0; cy = 50.0; near = 0.001; far = 16.0
        return b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

    ### Constructors ###

    # Generic loading from FeatureTrackData
    @classmethod
    def task_from_feature_track_data(
        cls, feature_track_data: b3d.io.FeatureTrackData,
        n_frames=None,
        min_pixeldist_between_keypoints=None
    ):
        ftd = feature_track_data
        if n_frames is None:
            n_frames = ftd.rgbd_images.shape[0]

        rgbds = ftd.rgbd_images

        if min_pixeldist_between_keypoints is None:
            H = rgbds.shape[1]
            min_pixeldist_between_keypoints = H // 80

        keypoint_bool_mask = ftd.keypoint_visibility[0]
        keypoint_positions_2D_frame0_unfiltered = ftd.observed_keypoints_positions[0, keypoint_bool_mask][:, ::-1]
        valid_indices = get_keypoint_filter(min_pixeldist_between_keypoints)(keypoint_positions_2D_frame0_unfiltered)
        keypoint_positions_2D_frame0 = keypoint_positions_2D_frame0_unfiltered[valid_indices]
        keypoint_positions_3D = ftd.latent_keypoint_positions[:n_frames, keypoint_bool_mask, ...][:, valid_indices, ...]
        renderer = b3d.Renderer.from_intrinsics_object(b3d.camera.Intrinsics.from_array(ftd.camera_intrinsics))
        
        # For now this class doesn't support changing camera pose, so just use the first frame camera pose.
        # TODO: support changing camera pose!
        X_WC = b3d.Pose(ftd.camera_position[0], ftd.camera_quaternion[0])

        return cls(
            rgbds[:n_frames], X_WC,
            keypoint_positions_2D_frame0, keypoint_positions_3D,
            renderer=renderer
        )
    
    ### A handful of unity scenes ###

    @classmethod
    def task_for_sliding_book_scene(cls, n_frames):
        path = os.path.join(
            b3d.get_assets_path(),
            "shared_data_bucket/input_data/unity/keypoints/indoorplant/slidingBooks_60fps_lit_bg_800p.input.npz"
        )
        ftd = b3d.io.FeatureTrackData.load(path).slice_time(start_frame=21)
        return cls.task_from_feature_track_data(ftd, n_frames)

    @classmethod
    def task_from_known_unity_scene_spec(cls, spec, **kwargs):
        ftd = b3d.io.FeatureTrackData.load(spec['path']).slice_time(start_frame=spec['start_frame'])
        return cls.task_from_feature_track_data(ftd, **kwargs)

    @classmethod
    def get_known_unity_scene_specs(cls):
        # These are filenames in 'shared_data_bucket/input_data/unity/keypoints/indoorplant/'
        good_filename_starttime_pairs = [
            ("plantRoomLookingThrough_30fps_lit_bg_800p.input.npz", 0),
            ("slidingBooks_60fps_lit_bg_800p.input.npz", 21),
            ("slidingPiledBooks_60fps_lit_bg_800p.input.npz", 21)
        ]
        return [
            {
                "scene_name": filename,
                "path": os.path.join(b3d.get_assets_path(), "shared_data_bucket/input_data/unity/keypoints/indoorplant", filename),
                "start_frame": starttime
            }
            for filename, starttime in good_filename_starttime_pairs
        ]
    
    # Scene manually constructed in Python: rotating cheezit box
    @classmethod
    def task_from_rotating_cheezit_box(cls, n_frames=30):
        (renderer, centers_2D_frame_0, centers_3D_W_over_time, X_WC, observed_rgbds) = cls.load_rotating_cheezit_box_data(n_frames)
        return cls(observed_rgbds, X_WC, centers_2D_frame_0, centers_3D_W_over_time, renderer=renderer)

    @classmethod
    def load_rotating_cheezit_box_data(cls, n_frames):
        renderer = cls.get_default_renderer()

        mesh_path = os.path.join(b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
        mesh = trimesh.load(mesh_path)
        cheezit_object_library = b3d.MeshLibrary.make_empty_library()
        cheezit_object_library.add_trimesh(mesh)
        box_poses_W = vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
        box_poses_W = b3d.Pose.from_matrix(box_poses_W)
        box_poses_W = box_poses_W[:n_frames]
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

        # Values are in pixel space, in order (H, W)
        width_gradations = jnp.arange(44, 84, 10) - 12
        height_gradations = jnp.arange(38, 90, 10) - 12
        centers_2D_frame_0 = all_pairs_2(height_gradations, width_gradations)
        
        centers_3D_frame0_C = xyzs_C[0][centers_2D_frame_0[:, 0], centers_2D_frame_0[:, 1]]
        centers_3D_frame0_W = X_WC.apply(centers_3D_frame0_C)

        # Let frame B0 be the first box pose
        X_W_B0 = box_poses_W[0]
        centers_3D_B0 = X_W_B0.inv().apply(centers_3D_frame0_W)
        centers_3D_W_over_time = jax.vmap(
            lambda X_W_Bt: X_W_Bt.apply(centers_3D_B0)
        )(box_poses_W)

        return (renderer, centers_2D_frame_0, centers_3D_W_over_time, X_WC, observed_rgbds)

### Utils ###

def all_pairs_2(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(X, Y), axis=-1),
        0, 1
    ).reshape(-1, 2)

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

### Filter 2D keypoints to ones that are sufficently distant ###
def get_keypoint_filter(max_pixel_dist):
    """
    Get a function that accepts a collection of 2D keypoints in pixel coordinates as input,
    and returns a list of indices of a subset of the keypoints, such that all the selected
    keypoints are at least `max_pixel_dist` pixels apart from each other.
    """
    def filter_step_i_if_valid(st):
        i, patch_positions_2D = st
        distances = jnp.linalg.norm(patch_positions_2D - patch_positions_2D[i], axis=-1)
        invalid_indices = jnp.logical_and(distances < max_pixel_dist, jnp.arange(patch_positions_2D.shape[0]) > i)
        patch_positions_2D = jnp.where(
            invalid_indices[:, None],
            -jnp.ones(2),
            patch_positions_2D,
        )
        return (i+1, patch_positions_2D)

    @jax.jit
    def filter_step_i(st):
        i, patch_positions_2D = st
        return jax.lax.cond(
            jnp.all(patch_positions_2D[i] == -jnp.ones(2)),
            lambda st: (st[0]+1, st[1]),
            filter_step_i_if_valid,
            st
        )

    def filter_patch_positions(patch_positions_2D):
        """
        Returns a list of 2D keypoints indices that have not been filtered out.
        """
        i = 0
        while i < patch_positions_2D.shape[0]:
            i, patch_positions_2D = filter_step_i((i, patch_positions_2D))

        return jnp.where(jnp.all(patch_positions_2D != -1., axis=-1))[0]
    
    return filter_patch_positions