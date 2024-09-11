from abc import abstractmethod
from typing import Mapping

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey

from b3d.chisight.gen3d.pixel_kernels import (
    FullPixelColorDistribution,
    FullPixelDepthDistribution,
    PixelDepthDistribution,
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

    near: float = Pytree.static()
    far: float = Pytree.static()
    image_height: int = Pytree.static()
    image_width: int = Pytree.static()

    def get_pixels_points_association(
        self, transformed_points, hyperparams: Mapping
    ) -> PixelsPointsAssociation:
        return PixelsPointsAssociation.from_points_and_intrinsics(
            transformed_points,
            hyperparams["intrinsics"],
            self.image_height,
            self.image_width,
        )

    @abstractmethod
    def sample(self, key: PRNGKey, state: Mapping, hyperparams: Mapping) -> FloatArray:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self, obseved_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
    ) -> FloatArray:
        raise NotImplementedError

    def get_depth_vertex_kernel(self) -> PixelDepthDistribution:
        raise NotImplementedError

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        raise NotImplementedError


@Pytree.dataclass
class NoOcclusionPerVertexImageKernel(ImageKernel):
    near: float = Pytree.static()
    far: float = Pytree.static()
    image_height: int = Pytree.static()
    image_width: int = Pytree.static()

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

        keys = jax.random.split(key, (self.image_height, self.image_width))
        return jax.vmap(
            jax.vmap(vertex_kernel.sample, in_axes=(0, 0, None, None, 0, 0)),
            in_axes=(0, 0, None, None, 0, 0),
        )(
            keys,
            pixel_latent_rgbd,
            state["color_scale"],
            state["depth_scale"],
            pixel_visibility_prob,
            pixel_depth_nonreturn_prob,
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

        scores = jax.vmap(vertex_kernel.logpdf, in_axes=(0, 0, None, None, 0, 0))(
            observed_rgbd_per_point,
            latent_rgbd_per_point,
            state["color_scale"],
            state["depth_scale"],
            state["visibility_prob"],
            state["depth_nonreturn_prob"],
        )
        # the pixel kernel does not expect invalid observed_rgbd and will return
        # -inf if it is invalid. We need to filter those out here.
        # (invalid rgbd could happen when the vertex is projected out of the image)
        scores = jnp.where(is_unexplained(observed_rgbd_per_point), 0.0, scores)

        return scores.sum()

    def get_rgbd_vertex_kernel(self) -> PixelRGBDDistribution:
        # Note: The distributions were originally defined for per-pixel computation,
        # but they should work for per-vertex computation as well, except that
        # they don't expect observed_rgbd to be invalid, so we need to handle
        # that manually.
        return PixelRGBDDistribution(
            FullPixelColorDistribution(),
            FullPixelDepthDistribution(self.near, self.far),
        )

    def get_depth_vertex_kernel(self) -> PixelDepthDistribution:
        return self.get_rgbd_vertex_kernel().depth_kernel
