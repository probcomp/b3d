from typing import NamedTuple, TypeAlias

import jax.numpy as jnp

from b3d.types import Array, Float, Int

ImageShape: TypeAlias = tuple[int, ...]
ScreenCoordinates: TypeAlias = Array
CameraCoordinates: TypeAlias = Array
DepthImage: TypeAlias = Array
CameraMatrix3x3: TypeAlias = Array
BoundingBox: TypeAlias = Array


class Intrinsics(NamedTuple):
    width: Int
    height: Int
    fx: Float
    fy: Float
    cx: Float
    cy: Float
    near: Float
    far: Float

    @classmethod
    def from_array(cls, intrinsics: Array):
        return cls(
            jnp.asarray(intrinsics[0], dtype=jnp.int32),
            jnp.asarray(intrinsics[1], dtype=jnp.int32),
            *intrinsics[2:],
        )

    def as_array(self):
        """Returns intrinsics as a float array."""
        return jnp.array(self)
    
    def downscale(self, factor):
        return Intrinsics(
            self.width // factor,
            self.height // factor,
            self.fx / factor,
            self.fy / factor,
            self.cx / factor,
            self.cy / factor,
            self.near,
            self.far,
        )


class RenderConfig(NamedTuple):
    bg_color: Array = jnp.array([1.0, 1.0, 1.0])


def pixel_centers_from_shape(img_shape: ImageShape) -> ScreenCoordinates:
    """
    Returns an array of sensor coordinates `uv` of the centers of each pixel of an image.

    Args:
        `img_shape`: (H,W) shape of an image.

    Returns:
        (H,W,2) array of sensor coordinates of the centers of each image pixel.
    """
    v, u = jnp.mgrid[: img_shape[0], : img_shape[1]]
    return jnp.stack([u, v], axis=-1) + 0.5


def camera_from_screen_and_depth(
    uv: ScreenCoordinates, z: DepthImage, intrinsics
) -> CameraCoordinates:
    """
    Returns camera coordinates `xyz` from sensor coordinates `uv`, and depth measurements `z`.
    These are related by the camera matrix $K$ as follows:
    $$
        (x, y, z)^T = K^{-1} * (z*u, z*v, z)^T.
    $$

    Args:
        `uv`: (...,2) array of  sensor coordinates.
        `z`:  (...,)  array of depth measurements.
        `intrinsics`: Intrinsics.

    Returns:
        (...,3) array of camera coordinates.
    """
    _, _, fx, fy, cx, cy, _, _ = intrinsics
    u, v = uv[..., 0], uv[..., 1]
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz


def camera_from_screen(uv: ScreenCoordinates, intrinsics) -> CameraCoordinates:
    z = jnp.ones(uv.shape[:-1])
    return camera_from_screen_and_depth(uv, z, intrinsics)


def camera_from_depth(z: DepthImage, intrinsics) -> CameraCoordinates:
    """
    Maps to camera coordinates `xyz` from depth measurements `z`.
    The relation between `xyz` and `z` is given by
    $$
        (x, y, z)^T = K^{-1} * (z*u, z*v, z)^T,
    $$
    where $(u,v)$ is the center of the pixel
    associated with the measurement $z$, and
    $K$ is the camera matrix.

    Args:
        `z`: (H,W) array of depth measurements.
        `intrinsics`: Intrinsics.

    Returns:
        (H,W,3) array of camera coordinates.
    """
    uv = pixel_centers_from_shape(z.shape)
    return camera_from_screen_and_depth(uv, z, intrinsics)


xyz_from_depth = camera_from_depth
unproject_depth = camera_from_depth


def screen_from_camera(xyz: CameraCoordinates, intrinsics, culling=False) -> ScreenCoordinates:
    """
    Maps to sensor coordintaes `uv` from camera coordinates `xyz`, which are
    defined by $(u,v) = (u'/z,v'/z)$, where
    $$
        (u', v', z)^T = K * (x, y, z)^T,
    $$
    and $K$ is the camera matrix.

    Args:
        `xyz`: (...,3) array of camera coordinates.
        `intrinsics`: Intrinsics.

    Returns:
        (...,2) array of screen coordinates.
    """
    _, _, fx, fy, cx, cy, near, far = intrinsics
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u_ = x * fx / z + cx
    v_ = y * fy / z + cy

    # TODO: What is the right way of doing this? Returning infs?
    in_range = ((near <= z) & (z <= far)) | (not culling)

    u = jnp.where(in_range, u_, jnp.inf)
    v = jnp.where(in_range, v_, jnp.inf)

    return jnp.stack([u, v], axis=-1)


screen_from_xyz = screen_from_camera


def screen_from_world(x, cam, intr, culling=False):
    """Maps to screen coordintaes `uv` from world coordinates `xyz`."""
    return screen_from_camera(cam.inv().apply(x), intr, culling=culling)

def world_from_screen(uv, cam, intr):
    """Maps to world coordintaes `xyz` from screen coords `uv`."""
    return cam.apply(camera_from_screen(uv, intr))

def camera_matrix_from_intrinsics(intr: Intrinsics) -> CameraMatrix3x3:
    """
    Returns a 3x3 camera matrix following the OpenCV convention, ie.
    ```
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    ```
    """
    return jnp.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]
    )


def camera_rays_from_intrinsics(intr: Intrinsics) -> CameraCoordinates:
    """
    Returns an camera coordinates for each center of a pixel on
    the embedded camera canvas.

    Args:
        `intr`: Intrinsics.
    Returns:
        (H, W, 3)-array of camera coordinates (where `H, W` are given in `intr`)
    """
    depth = jnp.ones((intr.height, intr.width))
    return xyz_from_depth(depth, intr)


def canvas_from_intrinsics(intr: Intrinsics) -> BoundingBox:
    """
    Returns the bounding box of the sensor canvas in camera space.

    Args:
        `intr`: Intrinsics.
    Returns:
        (2, 3)-array of camera coordinates, encoding
        lower-left and upper-right corners of the bounding box.
    """
    uv_bounds = jnp.array([[0.0, 0.0], [intr.width, intr.height]])
    xyz_bounds = camera_from_screen_and_depth(uv_bounds, jnp.array(1.0), intr)
    return xyz_bounds


def homogeneous_coordinates(xs, z=jnp.array(1.0)):
    """
    Maps from planar to homogeneous coordinates, eg.,
    maps (x,y) to (x, y, 1).

    Args:
        `xs`: (...,N) array of N-dim points.

    Returns:
        (...,N+1) array of homogeneous coordinates.
    """
    return jnp.concatenate([xs, jnp.ones_like(xs[..., :1])], axis=-1) * z[..., None]

homogeneous = homogeneous_coordinates


def planar_coordinates(xs):
    """
    Maps homogeneous to planar coordinates, eg.,
    maps (x,y,z) to (x/z, y/z).

    Args:
        `xs`: (...,N+1) array of N+1-dim points.

    Returns:
        (...,N) array of planar coordinates.
    """
    return xs[..., :-1] / xs[..., -1:]


def rgb_for_point_from_img(xyz, img_rgb, intrinsics_rgb):
    """
    Projects a point onto an image and returns the color.

    Args:
        `xyz`: (..., 3) array of 3D points
        `img_rgb`: (H, W, 3) array of colors (color image)
        `intrinsics_rgb`: Intrinsics

    Returns:
        (..., 3)-array of colors for each point in xyz
    """
    uv = screen_from_xyz(xyz, intrinsics_rgb)

    # TODO: resolve out of bounds with default color
    w, h = intrinsics_rgb.width, intrinsics_rgb.height
    uv = jnp.clip(uv, jnp.array([0, 0]), jnp.array([w, h]))
    ji = jnp.floor(uv).astype(jnp.int32)
    return img_rgb[ji[..., 1], ji[..., 0], :]


CAM_ALONG_X = jnp.array(
    [
        [0, 0, 1, -1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
)
