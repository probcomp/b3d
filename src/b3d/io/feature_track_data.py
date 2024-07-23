from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from b3d.camera import Intrinsics
from b3d.pose import Pose
from b3d.types import Array, Float
from b3d.utils import downsize_images

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
    fps: Optional[Float] = None
    observed_features: Optional[Array] = None
    latent_keypoint_positions: Optional[Array] = None
    latent_keypoint_quaternions: Optional[Array] = None
    object_assignments: Optional[Array] = None
    camera_position: Optional[Array] = None
    camera_quaternion: Optional[Array] = None

    def __init__(
        self,
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
        camera_quaternion: Optional[Array] = None,
    ):
        if rgbd_images.shape[-1] == 3:
            rgbd_images = jnp.concatenate(
                [rgbd_images, jnp.zeros(rgbd_images.shape[:-1] + (1,))], axis=-1
            )

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

    def flip_xy(self):
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[..., ::-1],
            observed_features=self.observed_features,
            keypoint_visibility=self.keypoint_visibility,
            rgbd_images=self.rgbd_images,
            latent_keypoint_positions=self.latent_keypoint_positions,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=self.camera_intrinsics,
        )

    @property
    def num_frames(self):
        return self.observed_keypoints_positions.shape[0]

    @property
    def num_keypoints(self):
        return self.observed_keypoints_positions.shape[1]

    @property
    def shape(self):
        return (self.num_frames, self.num_keypoints)

    @property
    def uv(self):
        return self.observed_keypoints_positions

    @property
    def vis(self):
        return self.visibility

    @property
    def rgb(self):
        return self.rgbd_images[..., :3]

    @property
    def visibility(self):
        return self.keypoint_visibility

    @property
    def rgbd(self):
        return self.rgbd_images

    @property
    def camera_poses(self):
        return Pose(self.camera_position, self.camera_quaternion)

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
            fps=self.fps,
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
            rgbd_images = get_or_none(data, "rgbd_images")
            if rgbd_images is None:
                rgb_imgs = get_or_none(data, "rgb_imgs")
                assert (
                    rgb_imgs is not None
                ), "One of rgbd_images or rgb_imgs must be provided."
                xyz = get_or_none(data, "xyz")
                if xyz is not None:
                    rgbd_images = jnp.concatenate([rgb_imgs, xyz[:, :, :, 2:]], axis=-1)
                else:
                    rgbd_images = jnp.concatenate(
                        [rgb_imgs, jnp.zeros(rgb_imgs.shape[:-1] + (1,))], axis=-1
                    )

            # Also handle loading from either the current `keypoint_visibility`
            # format, or from the older `latent_keypoint_visibility` format.
            keypoint_visibility = get_or_none(data, "keypoint_visibility")
            if keypoint_visibility is None:
                keypoint_visibility = get_or_none(data, "latent_keypoint_visibility")

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
                fps=get_or_none(data, "fps"),
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
            observed_keypoints_positions=self.observed_keypoints_positions[
                start_frame:end_frame
            ],
            observed_features=self.observed_features[start_frame:end_frame]
            if self.observed_features is not None
            else None,
            keypoint_visibility=self.keypoint_visibility[start_frame:end_frame],
            rgbd_images=self.rgbd_images[start_frame:end_frame],
            latent_keypoint_positions=self.latent_keypoint_positions[
                start_frame:end_frame
            ]
            if self.latent_keypoint_positions is not None
            else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[
                start_frame:end_frame
            ]
            if self.latent_keypoint_quaternions is not None
            else None,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position[start_frame:end_frame]
            if self.camera_position is not None
            else None,
            camera_quaternion=self.camera_quaternion[start_frame:end_frame]
            if self.camera_quaternion is not None
            else None,
            camera_intrinsics=self.camera_intrinsics,
            fps=self.fps,
        )

    def __len__(self):
        return self.observed_keypoints_positions.shape[0]

    def __getitem__(self, k):
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[k],
            observed_features=self.observed_features[k]
            if self.observed_features is not None
            else None,
            keypoint_visibility=self.keypoint_visibility[k],
            rgbd_images=self.rgbd_images[k],
            latent_keypoint_positions=self.latent_keypoint_positions[k]
            if self.latent_keypoint_positions is not None
            else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[k]
            if self.latent_keypoint_quaternions is not None
            else None,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position[k]
            if self.camera_position is not None
            else None,
            camera_quaternion=self.camera_quaternion[k]
            if self.camera_quaternion is not None
            else None,
            camera_intrinsics=self.camera_intrinsics,
            fps=self.fps,
        )

    def downscale(self, factor):
        """Downscales the rgbd images by the given factor."""
        assert (
            factor >= 1 and int(factor) == factor
        ), "Factor must be an integer greater than or equal to 1."
        factor = int(factor)
        if factor == 1:
            return self

        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions / factor,
            observed_features=self.observed_features,
            keypoint_visibility=self.keypoint_visibility,
            # TODO: Improve the downscaling procedure for the video.
            rgbd_images=self.rgbd_images[:, ::factor, ::factor, :],
            latent_keypoint_positions=self.latent_keypoint_positions,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions,
            object_assignments=self.object_assignments,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=Intrinsics.from_array(self.intrinsics)
            .downscale(factor)
            .as_array(),
        )

    def slice_keypoints(self, keypoint_indices):
        """Returns a FeatureTrackData with only the selected keypoints."""
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[
                :, keypoint_indices, :
            ],
            keypoint_visibility=self.keypoint_visibility[:, keypoint_indices],
            camera_intrinsics=self.camera_intrinsics,
            rgbd_images=self.rgbd_images,
            fps=self.fps if self.fps is not None else None,
            observed_features=self.observed_features[:, keypoint_indices, :]
            if self.observed_features is not None
            else None,
            latent_keypoint_positions=self.latent_keypoint_positions[
                :, keypoint_indices, :
            ]
            if self.latent_keypoint_positions is not None
            else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[
                :, keypoint_indices, :
            ]
            if self.latent_keypoint_quaternions is not None
            else None,
            object_assignments=self.object_assignments[keypoint_indices]
            if self.object_assignments is not None
            else None,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
        )

    def has_depth_channel(self):
        return jnp.any(self.rgbd_images[..., 3] > 0.0)

    def remove_points_invisible_at_frame0(self):
        """
        Filter this FeatureTrackData to only include keypoints that are visible at frame 0.
        """
        mask = self.keypoint_visibility[0]
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[:, mask],
            observed_features=self.observed_features[:, mask]
            if self.observed_features is not None
            else None,
            keypoint_visibility=self.keypoint_visibility[:, mask],
            rgbd_images=self.rgbd_images,
            latent_keypoint_positions=self.latent_keypoint_positions[:, mask]
            if self.latent_keypoint_positions is not None
            else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[:, mask]
            if self.latent_keypoint_quaternions is not None
            else None,
            object_assignments=self.object_assignments[mask]
            if self.object_assignments is not None
            else None,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=self.camera_intrinsics,
        )

    def all_points_visible_at_frame0(self):
        return jnp.all(self.keypoint_visibility[0])

    def sparsify_points_to_minimum_2D_distance_at_frame0(self, min_2D_distance):
        """
        Filter the keypoints in this FeatureTrackData object, returning a new
        FeatureTrackData in which no two observed keypoints
        are closer than `min_2D_distance` pixels in the first frame.
        """
        assert self.all_points_visible_at_frame0(), "In the current implementation, all keypoints must be visible at frame 0 to filter by minimimum 2D distance."
        valid_indices = get_keypoint_filter(min_2D_distance)(
            self.observed_keypoints_positions[0]
        )
        return FeatureTrackData(
            observed_keypoints_positions=self.observed_keypoints_positions[
                :, valid_indices, :
            ],
            observed_features=self.observed_features[:, valid_indices]
            if self.observed_features is not None
            else None,
            keypoint_visibility=self.keypoint_visibility[:, valid_indices],
            rgbd_images=self.rgbd_images,
            latent_keypoint_positions=self.latent_keypoint_positions[:, valid_indices]
            if self.latent_keypoint_positions is not None
            else None,
            latent_keypoint_quaternions=self.latent_keypoint_quaternions[
                :, valid_indices
            ]
            if self.latent_keypoint_quaternions is not None
            else None,
            object_assignments=self.object_assignments[valid_indices]
            if self.object_assignments is not None
            else None,
            camera_position=self.camera_position,
            camera_quaternion=self.camera_quaternion,
            camera_intrinsics=self.camera_intrinsics,
        )

    def min_2D_distance_at_frame0(self):
        """
        Returns the minimum 2D distance between any two keypoints at frame 0.
        """
        distances = jnp.linalg.norm(
            self.observed_keypoints_positions[None]
            - self.observed_keypoints_positions[:, None],
            axis=-1,
        )
        distances = jnp.where(jnp.eye(distances.shape[0]) == 1.0, jnp.inf, distances)
        return jnp.min(distances)

    def quick_plot(self, t=None, fname=None, ax=None, figsize=(3, 3), downsize=10):
        if t is None:
            figsize = (figsize[0] * self.num_frames, figsize[1])

        if ax is None:
            _fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.set_aspect(1)
            ax.axis("off")

        rgb = downsize_images(self.rgb, downsize)
        if t is None:
            _h, w = self.rgb.shape[1:3]
            ax.imshow(np.concatenate(rgb / 255, axis=1))
            ax.scatter(
                *np.concatenate(
                    [
                        self.uv[t, self.vis[t]] / downsize
                        + np.array([t * w, 0]) / downsize
                        for t in range(self.num_frames)
                    ]
                ).T,
                s=1,
            )
        else:
            ax.imshow(rgb[t] / 255)
            ax.scatter(*(self.uv[t, self.vis[t]] / downsize).T, s=1)


### Filter 2D keypoints to ones that are sufficently distant ###
def get_keypoint_filter(max_pixel_dist):
    """
    Get a function that accepts a collection of 2D keypoints in pixel coordinates as input,
    and returns a list of indices of a subset of the keypoints, such that all the selected
    keypoints are at least `max_pixel_dist` pixels apart from each other.
    """

    def filter_step_i_if_valid(st):
        i, keypoint_positions_2D = st
        distances = jnp.linalg.norm(
            keypoint_positions_2D - keypoint_positions_2D[i], axis=-1
        )
        invalid_indices = jnp.logical_and(
            distances < max_pixel_dist, jnp.arange(keypoint_positions_2D.shape[0]) > i
        )
        keypoint_positions_2D = jnp.where(
            invalid_indices[:, None],
            -jnp.ones(2),
            keypoint_positions_2D,
        )
        return (i + 1, keypoint_positions_2D)

    @jax.jit
    def filter_step_i(st):
        i, keypoint_positions_2D = st
        return jax.lax.cond(
            jnp.all(keypoint_positions_2D[i] == -jnp.ones(2)),
            lambda st: (st[0] + 1, st[1]),
            filter_step_i_if_valid,
            st,
        )

    def filter_keypoint_positions(keypoint_positions_2D):
        """
        Returns a list of 2D keypoints indices that have not been filtered out.
        """
        i = 0
        while i < keypoint_positions_2D.shape[0]:
            i, keypoint_positions_2D = filter_step_i((i, keypoint_positions_2D))

        return jnp.where(jnp.all(keypoint_positions_2D != -1.0, axis=-1))[0]

    return filter_keypoint_positions
