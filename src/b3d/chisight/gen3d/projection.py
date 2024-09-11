from functools import cached_property

import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, IntArray

import b3d.utils

# using this in combination with mode="drop" in the .at[]
# methods can help filter out vertices that are not visible in the image
INVALID_IDX = jnp.iinfo(jnp.int32).min  # -2147483648


@Pytree.dataclass
class PixelsPointsAssociation(Pytree):
    """A utility class to associate 3D points with their projected 2D pixel."""

    projected_pixel_coordinates: IntArray  # (num_vertices, 2)
    image_height: int
    image_width: int

    @classmethod
    def from_hyperparams_and_pose(cls, hyperparams, pose_CO):
        """`pose_CO` is the same thing as `pose` in the model."""
        vertices_O = hyperparams["vertices"]
        vertices_C = pose_CO.apply(vertices_O)
        return cls.from_points_and_intrinsics(
            vertices_C,
            hyperparams["intrinsics"],
            hyperparams["image_height"].const,
            hyperparams["image_width"].const,
        )

    @classmethod
    def from_points_and_intrinsics(
        cls,
        points: FloatArray,
        intrinsics: dict,
        image_height: int,
        image_width: int,
    ) -> "PixelsPointsAssociation":
        """Create a PixelsPointsAssociation object from a set of 3D points and
        the camera intrinsics.

        Args:
            points (FloatArray): The points/vertices in 3D space (num_vertices, 3).
            intrinsics (dict): Camera intrinsics.
            image_height (int): Height of the image.
            image_width (int): Width of the image.
        """
        projected_coords = jnp.rint(
            b3d.utils.xyz_to_pixel_coordinates(
                points,
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
            )
            - 0.5
        )
        # handle NaN before converting to int (otherwise NaN will be converted
        # to 0)
        projected_coords = jnp.nan_to_num(projected_coords, nan=INVALID_IDX)

        # handle the case where the projected coordinates are outside the image
        projected_coords = jnp.where(
            projected_coords > 0, projected_coords, INVALID_IDX
        )
        projected_coords = jnp.where(
            projected_coords < jnp.array([image_height, image_width]),
            projected_coords,
            INVALID_IDX,
        )

        return cls(projected_coords.astype(jnp.int32), image_height, image_width)

    def __len__(self) -> int:
        return self.projected_pixel_coordinates.shape[0]

    def shape(self) -> tuple[int, int]:
        return self.projected_pixel_coordinates.shape

    @property
    def x(self) -> IntArray:
        return self.projected_pixel_coordinates[:, 0]

    @property
    def y(self) -> IntArray:
        return self.projected_pixel_coordinates[:, 1]

    def get_point_rgbds(self, rgbd_image: FloatArray) -> FloatArray:
        """
        Get a (num_vertices, 4) array of RGBD values for each vertex
        by indexing into the given image.
        Vertices that don't hit a pixel will have a value of (-1, -1, -1, -1).
        """
        return rgbd_image.at[self.x, self.y].get(mode="drop", fill_value=-1.0)

    def get_point_depths(self, rgbd_image: FloatArray) -> FloatArray:
        """
        Get a (num_vertices,) array of depth values for each vertex
        by indexing into the given image, or -1 if the vertex doesn't hit a pixel.
        """
        return self.get_point_rgbds(rgbd_image)[..., 3]

    def get_point_rgbs(self, rgbd: FloatArray) -> FloatArray:
        """
        Get a (num_vertices, 3) array of RGB values for each vertex
        by indexing into the given image, or [-1, -1, -1] if the vertex doesn't hit a pixel.
        """
        return self.get_point_rgbds(rgbd)[..., :3]

    @cached_property
    def num_point_per_pixel(self) -> IntArray:
        """Return a 2D array of shape (image_height, image_width) where each
        element is the number of points that project to that pixel.
        """
        counts = jnp.zeros((self.image_height, self.image_width), dtype=jnp.int32)
        counts = counts.at[self.x, self.y].add(1, mode="drop")
        return counts

    @cached_property
    def pixel_to_point_idx(self) -> IntArray:
        """Return a 2D array of shape (image_height, image_width) where each
        element is the index of the point that projects to that pixel (if any).
        If none of the points project to that pixel, the value is set to INVALID_IDX.

        Warning: this implementaion does not handle race condition. That is, if
        multiple points project to the same pixel, this method will randomly
        return one of them (the non-determinism is subject to GPU parallelism).
        """
        registered_pixel_idx = jnp.full(
            (self.image_height, self.image_width), INVALID_IDX, dtype=jnp.int32
        )
        registered_pixel_idx = registered_pixel_idx.at[self.x, self.y].set(
            jnp.arange(len(self))
        )
        return registered_pixel_idx

    def get_pixel_idx(self, point_idx: int) -> IntArray:
        return self.projected_pixel_coordinates[point_idx]

    def pixels_with_multiple_points(self) -> tuple[IntArray, IntArray]:
        """Return a tuple of (x_coords, y_coords) of pixels that have more than
        one vertices associated with them. Note that this method is not JIT-compatible
        because the return values are not of fixed shape.
        """
        return jnp.nonzero(self.num_point_per_pixel > 1)

    def get_one_latent_point_idx(self, pixel_x: int, pixel_y: int) -> int:
        """Return the index of one of the points that project to the given pixel.
        If there are multiple points, this method will return one of them randomly
        (the non-determinism is subject to GPU parallelism).
        """
        return self.pixel_to_point_idx[pixel_x, pixel_y]
