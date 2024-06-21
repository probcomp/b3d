from dataclasses import dataclass
from b3d.types import Array
import jax.numpy as jnp
from b3d.camera import Intrinsics
from b3d.pose import Pose
from typing import Optional



DESCR = """
FeatureTrackData:
    Timesteps: {data.uv.shape[0]}
    Num Keypoints: {data.uv.shape[1]}
    Sensor shape (width x height): {data.rgb.shape[2]} x {data.rgb.shape[1]}
"""

@dataclass
class FeatureTrackData:
    """
    Feature track data class. Note: Spatial units are measured in meters.

    Args:
            observed_keypoints_positions: (T, N, 2) Float Array
            keypoint_visibility:          (T, N) Boolean Array
            camera_intrinsics:            (8,) Float Array of camera intrinsics, see `camera.py`.
            rgbd_images:                  (T, H, W, 4) Float Array
            (optional) observed_features :           (T, N, F) Float Array OR None
            (optional) latent_keypoint_positions :   (T, N, 3) Float Array OR None
            (optional) latent_keypoint_quaternions : (T, N, 4) Float Array OR None
            (optional) object_assignments:           (N,) Int Array OR None
            (optional) camera_position:              (T, 3) Float Array OR None
            (optional) camera_quaternion:            (T, 4) Float Array OR None
            
    Example:
    ```
    # Initialize
    data = FeatureTrackData(
        observed_keypoints_positions = uv,
        keypoint_visibility = vis,
        camera_intrinsics = intr,
        rgbd_images = rgb_or_rgbd)

    # Save and load
    fname = "_temp.npz"
    data.save(fname)
    data = FeatureTrackData.load(fname)

    # Quick access
    uv   = data.uv
    vis  = data.vis
    rgb  = data.rgb
    rgbd = data.rgbd
    intr = data.intrinsics
    cams = data.camera_poses
    ```

    FeatureTrackData.load also supports
    (1) Loading from npz files with a `rgb_imgs` field instead of `rgbd_images`.
        In this case, the loader also looks for a `xyz` field to complete the `rgbd_images`;
        if none is found, the depth channel is set to zeros.
    (2) Loading from npz files with a `latent_keypoint_visibility` field instead of `keypoint_visibility`.
    """
    # Required fields: Observed Data
    observed_keypoints_positions: Array
    keypoint_visibility: Array
    camera_intrinsics: Array
    rgbd_images: Array
    # Optional fields: Ground truth data
    observed_features: Optional[Array]
    latent_keypoint_positions: Optional[Array]
    latent_keypoint_quaternions: Optional[Array]
    object_assignments: Optional[Array]
    camera_position: Optional[Array]
    camera_quaternion: Optional[Array]

    def __init__(self,
                observed_keypoints_positions: Array,
                keypoint_visibility: Array,
                rgbd_images: Array,
                camera_intrinsics: Array,
                observed_features: Optional[Array] = None,
                latent_keypoint_positions: Optional[Array] = None,
                latent_keypoint_quaternions: Optional[Array] = None,
                object_assignments: Optional[Array] = None,
                camera_position: Optional[Array] = None,
                camera_quaternion: Optional[Array] = None):

        if rgbd_images.shape[-1] == 3:
            rgbd_images = jnp.concatenate([rgbd_images, jnp.zeros(rgbd_images.shape[:-1] + (1,))], axis=-1)

        self.observed_keypoints_positions = observed_keypoints_positions
        self.observed_features = observed_features
        self.keypoint_visibility = keypoint_visibility
        self.camera_intrinsics = camera_intrinsics
        self.rgbd_images = rgbd_images
        self.latent_keypoint_positions = latent_keypoint_positions
        self.latent_keypoint_quaternions = latent_keypoint_quaternions
        self.object_assignments = object_assignments
        self.camera_position = camera_position
        self.camera_quaternion = camera_quaternion

    @property
    def uv(self): return self.observed_keypoints_positions

    @property
    def visibility(self): return self.keypoint_visibility

    @property
    def vis(self): return self.visibility

    @property
    def rgb(self): return self.rgbd_images[...,:3]

    @property
    def rgbd(self): return self.rgbd_images

    @property
    def camera_poses(self): return Pose(self.camera_position, self.camera_quaternion)

    @property
    def intrinsics(self):
        """Returns camera intrinsics as an Intrinsics object"""
        return Intrinsics.from_array(self.camera_intrinsics)

    def __str__(self):
        return DESCR.format(data=self)

    def __repr__(self):
        return self.__str__()

    def save(self, filepath: str):
        """Saves input to file"""
        to_save = dict(
            latent_keypoint_positions=self.latent_keypoint_positions,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions,
            observed_keypoints_positions=self.observed_keypoints_positions,
            observed_features=self.observed_features,
            rgbd_images=self.rgbd_images,
            keypoint_visibility=self.keypoint_visibility,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=self.camera_intrinsics,
        )
        to_save = {k: v for k, v in to_save.items() if v is not None}
        jnp.savez(filepath, **to_save)

    @classmethod
    def load(cls, filepath: str):
        """Loads input from file"""
        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=True)

            def get_or_none(data, key):
                if key in data:
                    return jnp.array(data[key])
                else:
                    return None

            # Handle loading from either the current `rgbd_images`
            # format, or from the older `rgb_images` + maybe `xyz` format.
            rgbd_images=get_or_none(data, "rgbd_images")
            if rgbd_images is None:
                rgb_imgs = get_or_none(data, "rgb_imgs")
                assert rgb_imgs is not None, "One of rgbd_images or rgb_imgs must be provided."
                xyz = get_or_none(data, "xyz")
                if xyz is not None:
                    rgbd_images = jnp.concatenate([rgb_imgs, xyz], axis=-1)
                else:
                    rgbd_images = jnp.concatenate([
                        rgb_imgs, jnp.zeros(rgb_imgs.shape[:-1] + (1,))
                    ], axis=-1)

            # Also handle loading from either the current `keypoint_visibility`
            # format, or from the older `latent_keypoint_visibility` format.
            keypoint_visibility=get_or_none(
                    data, "keypoint_visibility"
                )
            if keypoint_visibility is None:
                keypoint_visibility = get_or_none(
                    data, "latent_keypoint_visibility"
                )

            return cls(
                latent_keypoint_positions=get_or_none(
                    data, "latent_keypoint_positions"
                ),
                latent_keypoint_quaternions=get_or_none(
                    data, "latent_keypoint_quaternions"
                ),
                observed_keypoints_positions=get_or_none(
                    data, "observed_keypoints_positions"
                ),
                observed_features=get_or_none(data, "observed_features"),
                rgbd_images=rgbd_images,
                keypoint_visibility=keypoint_visibility,
                object_assignments=get_or_none(data, "object_assignments"),
                camera_position=get_or_none(data, "camera_position"),
                camera_quaternion=get_or_none(data, "camera_quaternion"),
                camera_intrinsics=get_or_none(data, "camera_intrinsics"),
            )

    def slice_time(self, start_frame=None, end_frame=None):
        """Slices data in time."""
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.rgbd_images.shape[0]
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[start_frame:end_frame],
            observed_features=self.observed_features[start_frame:end_frame] if self.observed_features is not None else None,
            keypoint_visibility=self.keypoint_visibility[start_frame:end_frame],
            rgbd_images=self.rgbd_images[start_frame:end_frame],
            latent_keypoint_positions=self.latent_keypoint_positions[start_frame:end_frame] if self.latent_keypoint_positions is not None else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[start_frame:end_frame] if self.latent_keypoint_quaternions is not None else None,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position[start_frame:end_frame] if self.camera_position is not None else None,
            camera_quaternion=self.camera_quaternion[start_frame:end_frame] if self.camera_quaternion is not None else None,
            camera_intrinsics=self.camera_intrinsics
        )