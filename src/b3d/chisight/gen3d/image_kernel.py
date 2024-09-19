from abc import abstractmethod
from functools import cached_property
from typing import Mapping

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, IntArray, PRNGKey

import b3d.utils
from b3d.chisight.gen3d.pixel_kernels.pixel_rgbd_kernels import (
    PixelRGBDDistribution,
    is_unexplained,
)

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
        )

    @classmethod
    def from_points_and_intrinsics(
        cls, points: FloatArray, intrinsics: dict
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

        image_height, image_width = (
            intrinsics["image_height"].unwrap(),
            intrinsics["image_width"].unwrap(),
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

    def get_pixel_attributes(self, point_attributes: FloatArray) -> FloatArray:
        """Given a (num_vertices, attribute_length) array of point attributes,
        return a (image_height, image_width, attribute_length) array of attributes.
        Pixels that don't hit a vertex will have a value filled with -1.
        """
        return point_attributes.at[self.pixel_to_point_idx].get(
            mode="drop", fill_value=-1
        )

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

    def get_pixels_with_multiple_points(self) -> tuple[IntArray, IntArray]:
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


@jax.jit
def get_latent_and_observed_correspondences(state, hyperparams, observed_rgbd):
    transformed_points = state["pose"].apply(hyperparams["vertices"])
    points_to_pixels = PixelsPointsAssociation.from_points_and_intrinsics(
        transformed_points, hyperparams["intrinsics"]
    )
    observed_rgbd_per_point = points_to_pixels.get_point_rgbds(observed_rgbd)
    latent_rgbd_per_point = jnp.concatenate(
        (state["colors"], transformed_points[..., 2, None]), axis=-1
    )
    return latent_rgbd_per_point, observed_rgbd_per_point


@jax.jit
def get_latent_rgb_image(state, hyperparams):
    transformed_points = state["pose"].apply(hyperparams["vertices"])
    ppa = PixelsPointsAssociation.from_points_and_intrinsics(
        transformed_points, hyperparams["intrinsics"]
    )
    latent_rgb_image = jnp.clip(ppa.get_pixel_attributes(state["colors"]), 0.0, 1.0)
    return latent_rgb_image


@Pytree.dataclass
class ImageKernel(genjax.ExactDensity):
    """An abstract class that defines the common interface for image kernels,
    which generates a new RGBD image from the current state, controlled by
    the hyperparameters.

    The support of the distribution is [0, 1]^3 x [near, far].
    """

    @abstractmethod
    def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self, obseved_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
    ) -> FloatArray:
        raise NotImplementedError

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        raise NotImplementedError


@Pytree.dataclass
class NoOcclusionPerVertexImageKernel(ImageKernel):
    rgbd_vertex_kernel: PixelRGBDDistribution

    def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
        """Generate latent RGBD image by projecting the vertices directly to the image
        plane, without checking for occlusions.
        """

        transformed_points = state["pose"].apply(hyperparams["vertices"])
        points_to_pixels = PixelsPointsAssociation.from_points_and_intrinsics(
            transformed_points,
            hyperparams["intrinsics"],
        )
        vertex_kernel = self.get_rgbd_vertex_kernel()

        # assuming that at most one vertex hit the pixel, we can convert the
        # per-vertex attributes into per-pixel attributes, then vmap the
        # RGBD pixel kernel over the pixels to generate the image.
        pixel_visibility_prob = points_to_pixels.get_pixel_attributes(
            state["visibility_prob"]
        )
        pixel_depth_nonreturn_prob = points_to_pixels.get_pixel_attributes(
            state["depth_nonreturn_prob"]
        )
        pixel_latent_rgb = points_to_pixels.get_pixel_attributes(state["colors"])
        pixel_latent_depth = points_to_pixels.get_pixel_attributes(
            transformed_points[..., 2]
        )
        pixel_latent_rgbd = jnp.concatenate(
            [pixel_latent_rgb, pixel_latent_depth[..., None]], axis=-1
        )

        keys = jax.random.split(
            key,
            (
                hyperparams["intrinsics"]["image_height"].unwrap(),
                hyperparams["intrinsics"]["image_width"].unwrap(),
            ),
        )
        return jax.vmap(
            jax.vmap(vertex_kernel.sample, in_axes=(0, 0, None, None, 0, 0, None)),
            in_axes=(0, 0, None, None, 0, 0, None),
        )(
            keys,
            pixel_latent_rgbd,
            state["color_scale"],
            state["depth_scale"],
            pixel_visibility_prob,
            pixel_depth_nonreturn_prob,
            hyperparams["intrinsics"],
        )

    def logpdf(
        self, observed_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
    ) -> FloatArray:
        transformed_points = state["pose"].apply(hyperparams["vertices"])
        ppa = PixelsPointsAssociation.from_points_and_intrinsics(
            transformed_points, hyperparams["intrinsics"]
        )
        observed_rgbd_per_point = ppa.get_point_rgbds(observed_rgbd)
        latent_rgbd_per_point = jnp.concatenate(
            (state["colors"], transformed_points[..., 2, None]), axis=-1
        )

        vertex_kernel = self.get_rgbd_vertex_kernel()
        scores = jax.vmap(vertex_kernel.logpdf, in_axes=(0, 0, None, None, 0, 0, None))(
            observed_rgbd_per_point,
            latent_rgbd_per_point,
            state["color_scale"],
            state["depth_scale"],
            state["visibility_prob"],
            state["depth_nonreturn_prob"],
            hyperparams["intrinsics"],
        )

        # Points that don't hit the camera plane should not contribute to the score.
        scores = jnp.where(is_unexplained(observed_rgbd_per_point), 0.0, scores)
        score_for_pixels_with_points = scores.sum()

        # a = jnp.unique(ppa.projected_pixel_coordinates, axis=0, size=30000, fill_value=-1)
        # num_pixels = ( a.sum(-1) >= 0).sum()

        # TODO: add scores for pixels that don't get a point
        return score_for_pixels_with_points

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        # Note: The distributions were originally defined for per-pixel computation,
        # but they should work for per-vertex computation as well, except that
        # they don't expect observed_rgbd to be invalid, so we need to handle
        # that manually.
        return self.rgbd_vertex_kernel


### Unique point per pixel image kernel ###


def calculate_latent_and_observed_correspondences(
    observed_rgbd: FloatArray,
    state: Mapping,
    hyperparams: Mapping,
):
    intrinsics = hyperparams["intrinsics"]
    image_height, image_width = (
        intrinsics["image_height"].unwrap(),
        intrinsics["image_width"].unwrap(),
    )

    transformed_points = state["pose"].apply(hyperparams["vertices"])

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
    projected_coords = jnp.where(projected_coords > 0, projected_coords, INVALID_IDX)
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

    observed_rgbd_per_point = observed_rgbd.at[
        unique_pixel_coordinates[:, 0], unique_pixel_coordinates[:, 1]
    ].get(mode="drop", fill_value=-1.0)

    # For each collided pixel, get the color and depth of the vertex that hit that pixel.
    latent_rgbd_per_point = jnp.concatenate(
        [
            state["colors"][sort_order][unique_indices],
            transformed_points_sorted[unique_indices, 2, None],
        ],
        axis=-1,
    )

    is_valid = (
        (unique_pixel_coordinates >= 0).all(axis=-1)
        * (observed_rgbd_per_point[..., 3] >= 0.0),
    )[0]
    return (
        observed_rgbd_per_point,
        latent_rgbd_per_point,
        is_valid,
        unique_pixel_coordinates,
        sort_order[unique_indices],
    )


@Pytree.dataclass
class UniquePixelsImageKernel(ImageKernel):
    rgbd_vertex_kernel: PixelRGBDDistribution

    def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
        # TODO: implement this
        return jnp.zeros(
            (
                hyperparams["intrinsics"]["image_height"].unwrap(),
                hyperparams["intrinsics"]["image_width"].unwrap(),
                4,
            )
        )

    @jax.jit
    def logpdf(
        self, observed_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
    ) -> FloatArray:
        intrinsics = hyperparams["intrinsics"]
        (
            observed_rgbd_per_point,
            latent_rgbd_per_point,
            is_valid,
            _,
            point_indices_for_observed_rgbds,
        ) = calculate_latent_and_observed_correspondences(
            observed_rgbd, state, hyperparams
        )

        # Score the collided pixels
        scores = jax.vmap(
            hyperparams["image_kernel"].get_rgbd_vertex_kernel().logpdf,
            in_axes=(0, 0, None, None, 0, 0, None),
        )(
            observed_rgbd_per_point,
            latent_rgbd_per_point,
            state["color_scale"],
            state["depth_scale"],
            state["visibility_prob"][point_indices_for_observed_rgbds],
            state["depth_nonreturn_prob"][point_indices_for_observed_rgbds],
            hyperparams["intrinsics"],
        )
        total_score_for_explained_pixels = jnp.where(is_valid, scores, 0.0).sum()

        # Score the pixels that don't have any vertices hitting them.
        number_of_pixels_with_no_hypothesis = (
            hyperparams["intrinsics"]["image_height"].unwrap()
            * hyperparams["intrinsics"]["image_width"].unwrap()
        ) - is_valid.sum()

        number_of_total_non_return_pixels = (observed_rgbd[..., 3] == 0.0).sum()
        number_of_non_return_pixels_with_latent_hypothesis = (
            observed_rgbd_per_point[..., 3] == 0.0
        ).sum()
        number_of_non_return_pixels_without_latent_hypothesis = (
            number_of_total_non_return_pixels
            - number_of_non_return_pixels_with_latent_hypothesis
        )

        # The pixels that have no vertices produce an observation uniformly at random.
        color_score = jnp.log(1 / 1.0**3)

        depth_score_return = jnp.log(
            1 - hyperparams["unexplained_depth_nonreturn_prob"]
        ) + jnp.log(1 / (intrinsics["far"] - intrinsics["near"]))
        depth_score_non_return = jnp.log(
            hyperparams["unexplained_depth_nonreturn_prob"]
        )

        total_score_for_unexplained_pixels = (
            number_of_pixels_with_no_hypothesis
            - number_of_non_return_pixels_without_latent_hypothesis
        ) * (
            color_score + depth_score_return
        ) + number_of_non_return_pixels_without_latent_hypothesis * (
            color_score + depth_score_non_return
        )
        # Final score is the sum of the log probabilities over all pixels, collided and not collided.
        return total_score_for_explained_pixels + total_score_for_unexplained_pixels

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        return self.rgbd_vertex_kernel
