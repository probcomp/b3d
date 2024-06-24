from tests.common.task import Task
import jax
import jax.numpy as jnp
import b3d
import rerun as rr

class FeatureTrackData_AllInitiallyVisible_Task(Task):
    """
    Base class for tasks constructed from a `b3d.io.FeatureTrackData`,
    in which all keypoints are visible in the first frame.

    This base class exposes a default task specification (via
    a default `get_task_specification` implementation,
    and a default implementation of `visualize_task`.

    Downstream classes should implement `Task.score` and `Task.assert_passing` methods,
    and may wish to override the `get_task_specification` and `visualize_task` methods.

    The default task specification contains:
    - video [RGB or RGBD video]
    - Xs_WC [camera pose in the world frame, per frame]
    - initial_keypoint_positions_2D [2D keypoint center positions at frame 0]
        (N, 2) array of 2D keypoint center positions at frame 0
        stored as (y, x) pixel coordinates
    - renderer [Renderer object containing camera intrinsics]

    The class additionally stores:
    - keypoint_positions_3D [3D keypoint center positions at each frame]
        (T, N, 3) array
    - object_assignments [Object assignments for each keypoint]

    Indexing in the `N` dimension in any of these arrays will index to the same keypoint.
    """
    def __init__(self,
        video,
        Xs_WC,
        initial_keypoint_positions_2D,
        keypoint_positions_3D,
        object_assignments=None,
        renderer=None
    ):
        self.video = video
        self.Xs_WC = Xs_WC
        self.initial_keypoint_positions_2D = initial_keypoint_positions_2D
        self.keypoint_positions_3D = keypoint_positions_3D
        self.object_assignments = object_assignments
        if renderer is None:
            renderer = self.get_default_renderer()
        self.renderer = renderer

        # TODO: store the full observed keypoints array.

    def get_task_specification(self):
        # TODO: maybe remove this and just have downstream tasks define it?
        return {
            "video": self.video,
            "Xs_WC": self.Xs_WC,
            "initial_keypoint_positions_2D": self.initial_keypoint_positions_2D,
            "renderer": self.renderer
        }

    def visualize_task(self):
        # TODO: provide flags for downstream tasks to control what is visualized.

        rr.log("/task/frame0", rr.Image(self.video[0, :, :, :3]), timeless=True)
        rr.log("/task/initial_keypoint_positions_2D",
               rr.Points2D(self.initial_keypoint_positions_2D[:, ::-1], colors=jnp.array([0., 1., 0.])), timeless=True
            )

        for i in range(self.keypoint_positions_3D.shape[0]):
            rr.set_time_sequence("frame", i)
            rr.log("/task/keypoint_positions_3D", rr.Points3D(
                self.keypoint_positions_3D[i], colors=jnp.array([0., 1., 0.]), radii = 0.003)
            )
            renderer = self.renderer
            rr.log("/task/camera", rr.Pinhole(
                focal_length=[float(renderer.fx), float(renderer.fy)],
                width=renderer.width,
                height=renderer.height,
                principal_point=jnp.array([renderer.cx, renderer.cy]),
            ))
            X_WC = self.Xs_WC[i]
            rr.log("/task/camera", rr.Transform3D(translation=X_WC.pos, mat3x3=X_WC.rot.as_matrix()))

            rr.log("/task/camera/rgb_observed", rr.Image(self.video[i, :, :, :3]))

            if self.video.shape[-1] == 4:
                rr.log("/task/camera/depth_observed", rr.DepthImage(self.video[i, :, :, 3]))
                # If video is RGBD, get the point cloud and visualize it in the 3D viewer
                r = self.renderer
                xyzs_C = b3d.utils.xyz_from_depth_vectorized(
                    self.video[i, :, :, 3], r.fx, r.fy, r.cx, r.cy
                )
                rgbs = self.video[i, :, :, :3]
                xyzs_W = X_WC.apply(xyzs_C)
                rr.log("/task/observed_pointcloud", rr.Points3D(
                    positions=xyzs_W.reshape(-1,3),
                    colors=rgbs.reshape(-1,3),
                    radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
                )

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
            min_pixeldist_between_keypoints = max(H // 80, 6)

        # Filter the FeatureTrackData to only have the keypoints visible at frame 0
        keypoint_bool_mask = ftd.keypoint_visibility[0]
        keypoint_positions_2D_frame0_unfiltered = ftd.observed_keypoints_positions[0, keypoint_bool_mask][:, :]

        # Further filter the FeatureTrackData so that none of the keypoints are too close to each other
        valid_indices = get_keypoint_filter(min_pixeldist_between_keypoints)(keypoint_positions_2D_frame0_unfiltered)
        keypoint_positions_2D_frame0 = keypoint_positions_2D_frame0_unfiltered[valid_indices]
        keypoint_positions_3D = ftd.latent_keypoint_positions[:n_frames, keypoint_bool_mask, ...][:, valid_indices, ...]
        
        # Finish constructing the class
        renderer = b3d.Renderer.from_intrinsics_object(b3d.camera.Intrinsics.from_array(ftd.camera_intrinsics))
        Xs_WC = b3d.Pose(ftd.camera_position, ftd.camera_quaternion)

        # TODO: filter and store the full observed keypoints array.

        return cls(
            rgbds[:n_frames], Xs_WC,
            keypoint_positions_2D_frame0, keypoint_positions_3D,
            object_assignments=ftd.object_assignments,
            renderer=renderer
        )
    
### Utils ###

### Filter 2D keypoints to ones that are sufficently distant ###
def get_keypoint_filter(max_pixel_dist):
    """
    Get a function that accepts a collection of 2D keypoints in pixel coordinates as input,
    and returns a list of indices of a subset of the keypoints, such that all the selected
    keypoints are at least `max_pixel_dist` pixels apart from each other.
    """
    def filter_step_i_if_valid(st):
        i, keypoint_positions_2D = st
        distances = jnp.linalg.norm(keypoint_positions_2D - keypoint_positions_2D[i], axis=-1)
        invalid_indices = jnp.logical_and(distances < max_pixel_dist, jnp.arange(keypoint_positions_2D.shape[0]) > i)
        keypoint_positions_2D = jnp.where(
            invalid_indices[:, None],
            -jnp.ones(2),
            keypoint_positions_2D,
        )
        return (i+1, keypoint_positions_2D)

    @jax.jit
    def filter_step_i(st):
        i, keypoint_positions_2D = st
        return jax.lax.cond(
            jnp.all(keypoint_positions_2D[i] == -jnp.ones(2)),
            lambda st: (st[0]+1, st[1]),
            filter_step_i_if_valid,
            st
        )

    def filter_keypoint_positions(keypoint_positions_2D):
        """
        Returns a list of 2D keypoints indices that have not been filtered out.
        """
        i = 0
        while i < keypoint_positions_2D.shape[0]:
            i, keypoint_positions_2D = filter_step_i((i, keypoint_positions_2D))

        return jnp.where(jnp.all(keypoint_positions_2D != -1., axis=-1))[0]
    
    return filter_keypoint_positions
    
