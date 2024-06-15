from dataclasses import dataclass
from b3d.types import Array
import jax.numpy as jnp
from b3d.camera import Intrinsics
from typing import Optional


@dataclass
class FeatureTrackData:
    """
    Feature track data class. Note: Spatial units are measured in meters.

    Args:
            latent_keypoint_positions:    (T, N, 3) Float Array OR None
            latent_keypoint_quaternions:  (T, N, 4) Float Array OR None
            observed_keypoints_positions: (T, N, 2) Float Array
            observed_features:            (T, N, F) Float Array
            rgb_imgs:                     (T, H, W, 3) Float Array
            latent_keypoint_visibility:   (T, N) Boolean Array OR None
            object_assignments:           (N,) Int Array OR None
            camera_position:              (T, 3) Float Array
            camera_quaternion:            (T, 4) Float Array
            camera_intrinsics:            (8,) Float Array of camera intrinsics, see `camera.py`.
    """

    latent_keypoint_positions: Array
    latent_keypoint_quaternions: Optional[Array]
    observed_keypoints_positions: Array
    observed_features: Array
    rgb_imgs: Array
    latent_keypoint_visibility: Array
    object_assignments: Array
    camera_position: Array
    camera_quaternion: Array
    camera_intrinsics: Array

    @property
    def uv(self): self.observed_keypoints_positions

    @property
    def visibility(self): self.latent_keypoint_visibility

    @property
    def rgb(self): self.latent_keypoint_positions

    @property
    def camera_poses(self): Pose(self.camera_position, self.camera_quaternion)

    @property
    def intrinsics(self):
        """Returns camera intrinsics as an Intrinsics object"""
        return Intrinsics.from_array(self.camera_intrinsics)

    def save(self, filepath: str):
        """Saves input to file"""
        to_save = dict(
            latent_keypoint_positions=self.latent_keypoint_positions,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions,
            observed_keypoints_positions=self.observed_keypoints_positions,
            observed_features=self.observed_features,
            rgb_imgs=self.rgb_imgs,
            latent_keypoint_visibility=self.latent_keypoint_visibility,
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
                rgb_imgs=get_or_none(data, "rgb_imgs"),
                latent_keypoint_visibility=get_or_none(
                    data, "latent_keypoint_visibility"
                ),
                object_assignments=get_or_none(data, "object_assignments"),
                camera_position=get_or_none(data, "camera_position"),
                camera_quaternion=get_or_none(data, "camera_quaternion"),
                camera_intrinsics=get_or_none(data, "camera_intrinsics"),
            )

