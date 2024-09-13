from abc import abstractmethod
from typing import Mapping

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey

from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import (
    RenormalizedLaplacePixelColorDistribution,
    UniformPixelColorDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
    RenormalizedLaplacePixelDepthDistribution,
    UniformPixelDepthDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_rgbd_kernels import (
    FullPixelRGBDDistribution,
    PixelRGBDDistribution,
    is_unexplained,
)
from b3d.chisight.gen3d.projection import PixelsPointsAssociation


@Pytree.dataclass
class ImageKernel(genjax.ExactDensity):
    """An abstract class that defines the common interface for image kernels,
    which generates a new RGBD image from the current state, controlled by
    the hyperparameters.

    The support of the distribution is [0, 1]^3 x [near, far].
    """

    def get_pixels_points_association(
        self, transformed_points, hyperparams: Mapping
    ) -> PixelsPointsAssociation:
        return PixelsPointsAssociation.from_points_and_intrinsics(
            transformed_points,
            hyperparams["intrinsics"],
            hyperparams["intrinsics"]["image_height"].const,
            hyperparams["intrinsics"]["image_width"].const,
        )

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
    def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
        """Generate latent RGBD image by projecting the vertices directly to the image
        plane, without checking for occlusions.
        """
        transformed_points = state["pose"].apply(hyperparams["vertices"])
        points_to_pixels = self.get_pixels_points_association(
            transformed_points, hyperparams
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
        pixel_latent_rgbd = points_to_pixels.get_pixel_attributes(state["colors"])
        pixel_latent_depth = points_to_pixels.get_pixel_attributes(
            transformed_points[..., 2]
        )
        pixel_latent_rgbd = jnp.concatenate(
            [pixel_latent_rgbd, pixel_latent_depth[..., None]], axis=-1
        )

        keys = jax.random.split(
            key,
            (
                hyperparams["intrinsics"]["image_height"].const,
                hyperparams["intrinsics"]["image_width"].const,
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
        points_to_pixels = self.get_pixels_points_association(
            transformed_points, hyperparams
        )
        vertex_kernel = self.get_rgbd_vertex_kernel()
        observed_rgbd_per_point = points_to_pixels.get_point_rgbds(observed_rgbd)
        latent_rgbd_per_point = jnp.concatenate(
            (state["colors"], transformed_points[..., 2, None]), axis=-1
        )

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

        # TODO: add scores for pixels that don't get a point

        return score_for_pixels_with_points

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        # Note: The distributions were originally defined for per-pixel computation,
        # but they should work for per-vertex computation as well, except that
        # they don't expect observed_rgbd to be invalid, so we need to handle
        # that manually.
        return FullPixelRGBDDistribution(
            RenormalizedLaplacePixelColorDistribution(),
            UniformPixelColorDistribution(),
            RenormalizedLaplacePixelDepthDistribution(),
            UniformPixelDepthDistribution(),
        )


# @Pytree.dataclass
# class OldNoOcclusionPerVertexImageKernel(ImageKernel):
#     @jax.jit
#     def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
#         return jnp.zeros(
#             (
#                 hyperparams["intrinsics"]["image_height"].const,
#                 hyperparams["intrinsics"]["image_width"].const,
#                 4,
#             )
#         )

#     @jax.jit
#     def logpdf(
#         self, observed_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
#     ) -> FloatArray:
#         return self.info_func(observed_rgbd, state, hyperparams)["scores"].sum()

#     def info_from_trace(self, trace):
#         return self.info_func(
#             trace.get_choices()["rgbd"],
#             trace.get_retval()["new_state"],
#             trace.get_args()[0],
#         )

#     def info_func(self, observed_rgbd, state, hyperparams):
#         transformed_points = state["pose"].apply(hyperparams["vertices"])
#         projected_pixel_coordinates = jnp.rint(
#             b3d.xyz_to_pixel_coordinates(
#                 transformed_points,
#                 hyperparams["intrinsics"]["fx"],
#                 hyperparams["intrinsics"]["fy"],
#                 hyperparams["intrinsics"]["cx"],
#                 hyperparams["intrinsics"]["cy"],
#             )
#         ).astype(jnp.int32)

#         observed_rgbd_masked = observed_rgbd[
#             projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1]
#         ]

#         color_visible_branch_score = jax.scipy.stats.laplace.logpdf(
#             observed_rgbd_masked[..., :3], state["colors"], state["color_scale"]
#         ).sum(axis=-1)
#         color_not_visible_score = jnp.log(1 / 1.0**3)
#         color_score = jnp.logaddexp(
#             color_visible_branch_score + jnp.log(state["visibility_prob"]),
#             color_not_visible_score + jnp.log(1 - state["visibility_prob"]),
#         )

#         depth_visible_branch_score = jax.scipy.stats.laplace.logpdf(
#             observed_rgbd_masked[..., 3],
#             transformed_points[..., 2],
#             state["depth_scale"],
#         )
#         depth_not_visible_score = jnp.log(1 / 1.0)
#         _depth_score = jnp.logaddexp(
#             depth_visible_branch_score + jnp.log(state["visibility_prob"]),
#             depth_not_visible_score + jnp.log(1 - state["visibility_prob"]),
#         )
#         is_depth_non_return = observed_rgbd_masked[..., 3] < 0.0001

#         non_return_probability = 0.05
#         depth_score = jnp.where(
#             is_depth_non_return, jnp.log(non_return_probability), _depth_score
#         )

#         lmbda = 0.5
#         scores = lmbda * color_score + (1.0 - lmbda) * depth_score
#         return {
#             "scores": scores,
#             "observed_rgbd_masked": observed_rgbd_masked,
#         }

#     def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
#         # Note: The distributions were originally defined for per-pixel computation,
#         # but they should work for per-vertex computation as well, except that
#         # they don't expect observed_rgbd to be invalid, so we need to handle
#         # that manually.
#         raise NotImplementedError
