from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from b3d.types import Array, Float


@dataclass
class VideoInput:
    """
    Video data input. Note: Spatial units are measured in meters.

    World Coordinates. The floor is x,y and up is z.
    Camera Pose. The camera pose should be interpretted as the z-axis pointing out of the camera,
        x-axis pointing to the right, and y-axis pointing down. This is the OpenCV convention.
    Quaternions. We follow scipy.spatial.transform.Rotation.from_quat which uses scalar-last (x, y, z, w)
    Camera Intrinsics. We store it as an array of shape (8,) containing width, height, fx, fy, cx, cy, near, far.
        The camera matrix is given by: $$ K = \\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \\end{bmatrix} $$
    Spatial units. Spatial units are measured in meters (if not indicated otherwise).

    **Attributes:**
    - rgb
        video_input['rgb'][t,i,j] contains RGB values in the interval [0,255] of pixel i,j at time t.
        Shape: (T,H,W,3) -- Note this might be different from the width and height of xyz
        Type: uint8, in [0,255]
    - xyz
        video_input['xyz'][t,i,j] is the 3d point associated with pixel i,j at time t in camera coordinates
        Shape: (T, H', W', 3) -- Note this might be different from the width and height of rgb
        Type: Float
    - camera_positions
        video_input['camera_positions'][t] is the position of the camera at time t
        Shape: (T, 3)
        Type: Float
    - camera_quaternions
        video_input['camera_quaternions'][t] is the quaternion (in xyzw format) representing the orientation of the camera at time t
        Shape: (T, 4)
        Type: Float
    - camera_intrinsics_rgb
        video_input['camera_intrinsics_rgb'][:] contains width, height, fx, fy, cx, cy, near, far. Width and height determine the shape of rgb above
        Shape: (8,)
        Type: Float
    - camera_intrinsics_depth
        video_input['camera_intrinsics_depth'][:] contains width, height, fx, fy, cx, cy, near, far. Width and height determine the shape of xyz above
        Shape: (8,)
        Type: Float
    - fps
        Frames per second of the video
        Type: Float

    **Note:**
    For compactness, rgb values are saved as uint8 values, however
    the output of the renderer is a float between 0 and 1. VideoInput
    stores uint8 colors, so please use the rgb_float property for
    compatibility.

    **Note:**
    The width and height of the `rgb` and `xyz` arrays may differ.
    Their shapes match the entries in `camera_intrinsics_rgb` and
    `camera_intrinsics_depth`, respectively. The latter was used
    to project the `depth` arrays to `xyz`.
    """

    rgb: Array  # [num_frames, height_rgb, width_rgb, 3]
    xyz: Array  # [num_frames, height_depth, width_depth, 3]
    camera_positions: Array  # [num_frames, 3]
    camera_quaternions: Array  # [num_frames, 4]
    camera_intrinsics_rgb: (
        Array  # [8,] (width_rgb, height_rgb, fx, fy, cx, cy, near, far)
    )
    camera_intrinsics_depth: (
        Array  # [8,] (width_depth, height_depth, fx, fy, cx, cy, near, far)
    )

    @property
    def z(self):
        if self.xyz is not None:
            return self.xyz[..., [3]]
        else:
            return jnp.zeros(self.rgb.shape[:-1] + (1,))

    @property
    def depth(self):
        return self.z

    @property
    def rgbd(self):
        return jnp.concatenate([self.rgb, self.depth], axis=-1)

    def __init__(
        self,
        rgb,
        xyz=None,
        camera_positions=None,
        camera_quaternions=None,
        camera_intrinsics_rgb=None,
        camera_intrinsics_depth=None,
        fps=None,
    ):
        self.rgb = rgb
        self.xyz = xyz
        self.camera_positions = camera_positions
        self.camera_quaternions = camera_quaternions
        self.camera_intrinsics_rgb = camera_intrinsics_rgb
        self.camera_intrinsics_depth = camera_intrinsics_depth
        self.fps = fps

    def __post_init__(self):
        super().__init__()
        assert self.rgb.shape[0] == self.xyz.shape[0]
        assert self.rgb.shape[1] == self.camera_intrinsics_rgb[1]
        assert self.rgb.shape[2] == self.camera_intrinsics_rgb[0]
        assert self.rgb.dtype == jnp.uint8
        assert len(self.xyz.shape) == 4
        assert len(self.rgb.shape) == 4
        assert self.rgb.shape[-1] == 3
        assert self.xyz.shape[-1] == 3

    def to_dict(self):
        return {
            "rgb": self.rgb,
            "xyz": self.xyz,
            "camera_positions": self.camera_positions,
            "camera_quaternions": self.camera_quaternions,
            "camera_intrinsics_rgb": self.camera_intrinsics_rgb,
            "camera_intrinsics_depth": self.camera_intrinsics_depth,
            "fps": self.fps,
        }

    def save(self, filepath: str):
        """Saves VideoInput to file"""
        jnp.savez(
            filepath,
            rgb=self.rgb,
            xyz=self.xyz,
            camera_positions=self.camera_positions,
            camera_quaternions=self.camera_quaternions,
            camera_intrinsics_rgb=self.camera_intrinsics_rgb,
            camera_intrinsics_depth=self.camera_intrinsics_depth,
            fps=self.fps,
        )

    def save_in_timeframe(self, filepath: str, start_t: int, end_t: int):
        """Saves new VideoInput containing data
        between a timeframe into file"""
        jnp.savez(
            filepath,
            rgb=self.rgb[start_t:end_t],
            xyz=self.xyz[start_t:end_t],
            camera_positions=self.camera_positions[start_t:end_t],
            camera_quaternions=self.camera_quaternions[start_t:end_t],
            camera_intrinsics_rgb=self.camera_intrinsics_rgb,
            camera_intrinsics_depth=self.camera_intrinsics_depth,
            fps=self.fps,
        )

    @classmethod
    def load(cls, filepath: str):
        """Loads VideoInput from file"""

        def jnp_array_or_none(x):
            if x is None or x[()] is None:
                return None
            else:
                return jnp.array(x)

        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=True)

            fps = data["fps"] if ("fps" in data) else None

            return cls(
                rgb=jnp.array(data["rgb"]),
                xyz=jnp_array_or_none(data["xyz"]),
                camera_positions=jnp_array_or_none(data["camera_positions"]),
                camera_quaternions=jnp_array_or_none(data["camera_quaternions"]),
                camera_intrinsics_rgb=jnp_array_or_none(data["camera_intrinsics_rgb"]),
                camera_intrinsics_depth=jnp_array_or_none(
                    data["camera_intrinsics_depth"]
                ),
                fps=fps,
            )

    @property
    def rgb_float(self):
        if self.rgb.dtype == jnp.uint8:
            return self.rgb / 255.0
        else:
            return self.rgb