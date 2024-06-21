### FeatureTrackData object from the hgps repo.
# TODO: unify this and `b3d.io.FeatureTrackData`

import dataclasses
from b3d.types import Array
import jax.numpy as jnp
from b3d.camera import Intrinsics
from typing import Optional


@dataclasses.dataclass(kw_only=True)
class HGPSFeatureTrackData:
    """
    Feature track data class. Note: Spatial units are measured in meters.

    Args:
            latent_keypoint_positions:    (T, N, 3) Float Array
            latent_keypoint_quaternions:  (T, N, 4) Float Array OR None
            observed_keypoints_positions: (T, N, 2) Float Array
            observed_features:            (T, N, F) Float Array
            rgb_imgs:                     (T, H, W, 3) Float Array
            xyz:                          (T, H, W, 3) Float Array
            latent_keypoint_visibility:   (T, N) Boolean Array OR None
            object_assignments:           (N,) Int Array
            camera_position:              (T, 3) Float Array
            camera_quaternion:            (T, 4) Float Array
            camera_intrinsics:            (8,) Float Array of camera intrinsics, see `camera.py`.
    """

    latent_keypoint_positions: Array
    latent_keypoint_quaternions: Optional[Array] = None
    observed_keypoints_positions: Array
    observed_features: Array
    rgb_imgs: Array
    xyz: Array
    latent_keypoint_visibility: Optional[Array] = None
    object_assignments: Array
    camera_position: Array
    camera_quaternion: Array
    camera_intrinsics: Array

    def save(self, filepath: str):
        """Saves input to file"""
        to_save = {k: v for k, v in dataclasses.asdict(self).items() if v is not None}
        jnp.savez(filepath, **to_save)

    @classmethod
    def load(cls, filepath: str, startframe=0, img_downsize=1):
        """Loads input from file"""
        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=True)
            latent_keypoint_positions = data["latent_keypoint_positions"][startframe:]
            if "latent_keypoint_quaternions" in data:
                latent_keypoint_quaternions = data["latent_keypoint_quaternions"][startframe:]
            else:
                latent_keypoint_quaternions = None
            observed_keypoints_positions = data["observed_keypoints_positions"][startframe:, :, :] / img_downsize
            observed_features = data["observed_features"][startframe:]
            rgb_imgs = data["rgb_imgs"][startframe:, ::img_downsize, ::img_downsize, :]
            xyz = data["xyz"][startframe:, ::img_downsize, ::img_downsize, :]
            latent_keypoint_visibility = data["latent_keypoint_visibility"][startframe:]
            object_assignments = data["object_assignments"]
            camera_position = data["camera_position"][startframe:]
            camera_quaternion = data["camera_quaternion"][startframe:]
            camera_intrinsics = data["camera_intrinsics"]
            return cls(
                latent_keypoint_positions=latent_keypoint_positions,
                latent_keypoint_quaternions=latent_keypoint_quaternions,
                observed_keypoints_positions=observed_keypoints_positions,
                observed_features=observed_features,
                rgb_imgs=rgb_imgs,
                xyz=xyz,
                latent_keypoint_visibility=latent_keypoint_visibility,
                object_assignments=object_assignments,
                camera_position=camera_position,
                camera_quaternion=camera_quaternion,
                camera_intrinsics=camera_intrinsics,
            )

    @property
    def intrinsics_rgb(self):
        """Returns the RGB camera intrinsics as an Intrinsics object"""
        return Intrinsics.from_array(self.camera_intrinsics)