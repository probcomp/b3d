from typing import Mapping

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import IntArray, PRNGKey

import b3d.utils
from b3d.chisight.gen3d.pixel_kernels.pixel_rgbd_kernels import (
    PixelRGBDDistribution,
)

# using this in combination with mode="drop" in the .at[]
# methods can help filter out vertices that are not visible in the image
INVALID_IDX = jnp.iinfo(jnp.int32).min  # -2147483648


class PixelsPointsAssociation(Pytree):
    observed_pixel_indices: IntArray

    def from_pose_intrinsics_vertices(pose, intrinsics, vertices):
        image_height, image_width = (
            intrinsics["image_height"].unwrap(),
            intrinsics["image_width"].unwrap(),
        )
        transformed_points = pose.apply(vertices)

        # Sort the vertices by depth.
        sort_order = jnp.argsort(transformed_points[..., 2])
        transformed_points_sorted = transformed_points[sort_order]

        # Project the vertices to the image plane.
        projected_coords = jnp.rint(
            b3d.utils.xyz_to_pixel_coordinates(
                transformed_points_sorted,
                intrinsics["fx"],
                intrinsics["fy"],
                intrinsics["cx"],
                intrinsics["cy"],
            )
            - 0.5
        ).astype(jnp.int32)
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

        # Compute the unique pixel coordinates and the indices of the first vertex that hit that pixel.
        unique_pixel_coordinates, unique_indices = jnp.unique(
            projected_coords,
            axis=0,
            return_index=True,
            size=projected_coords.shape[0],
            fill_value=INVALID_IDX,
        )

        # Reorder the unique pixel coordinates, to the original point array indexing scheme
        observed_pixel_coordinates_per_point = -jnp.ones(
            (transformed_points.shape[0], 2), dtype=jnp.int32
        )
        observed_pixel_coordinates_per_point = observed_pixel_coordinates_per_point.at[
            sort_order[unique_indices]
        ].set(unique_pixel_coordinates)

        return PixelsPointsAssociation(observed_pixel_coordinates_per_point)

    def get_pixel_index(self, point_index):
        return self.observed_pixel_indices[point_index]


@Pytree.dataclass
class UniquePixelsImageKernel(genjax.ExactDensity):
    rgbd_vertex_kernel: PixelRGBDDistribution

    def sample(self, key: PRNGKey, ppa: PixelsPointsAssociation, state: Mapping, hyperparams: Mapping):
        return jax.vmap(
            jax.vmap(
                lambda i, j: self.rgbd_vertex_kernel.sample(
                    key,
                )
            )
        )
