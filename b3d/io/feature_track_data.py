from dataclasses import dataclass
from b3d.types import Array, Float
import jax.numpy as jnp
from b3d.camera import Intrinsics
from b3d.pose import Pose
from typing import Optional
import matplotlib.pyplot as plt


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
            (optional) fps:                          Float
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
    """
    # Required fields: Observed Data
    observed_keypoints_positions: Array
    keypoint_visibility: Array
    camera_intrinsics: Array
    rgbd_images: Array
    # Optional fields: Ground truth data
    fps: Optional[Float] = None
    observed_features: Optional[Array] = None
    latent_keypoint_positions: Optional[Array] = None
    latent_keypoint_quaternions: Optional[Array] = None
    object_assignments: Optional[Array] = None
    camera_position: Optional[Array] = None
    camera_quaternion: Optional[Array] = None

    def __init__(self,
                observed_keypoints_positions: Array,
                keypoint_visibility: Array,
                rgbd_images: Array,
                camera_intrinsics: Array,
                fps: Optional[Array] = None,
                observed_features: Optional[Array] = None,
                latent_keypoint_positions: Optional[Array] = None,
                latent_keypoint_quaternions: Optional[Array] = None,
                object_assignments: Optional[Array] = None,
                camera_position: Optional[Array] = None,
                camera_quaternion: Optional[Array] = None):

        if rgbd_images.shape[-1] == 3:
            rgbd_images = jnp.concatenate([rgbd_images, jnp.zeros(rgbd_images.shape[:-1] + (1,))], axis=-1)

        self.fps = fps
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
            fps = self.fps,
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
                rgbd_images=get_or_none(data, "rgbd_images"),
                keypoint_visibility=get_or_none(
                    data, "keypoint_visibility"
                ),
                fps = get_or_none(data, "fps"),
                object_assignments=get_or_none(data, "object_assignments"),
                camera_position=get_or_none(data, "camera_position"),
                camera_quaternion=get_or_none(data, "camera_quaternion"),
                camera_intrinsics=get_or_none(data, "camera_intrinsics"),
            )

    def quick_plot(self, t=0, ax=None, figsize=(3,3)):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_aspect(1)
            ax.axis("off")

        ax.imshow(self.rgb[t]/255)
        ax.scatter(*self.uv[t, self.vis[t]].T, s=1)