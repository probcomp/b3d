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
    PixelRGBDDistribution,
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
            hyperparams,  # should include fx, fy, cx, cy
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
        # TODO: to be finished...
        return jnp.zeros((self.image_height, self.image_width, 4))

    def logpdf(
        self, observed_rgbd: FloatArray, state: Mapping, hyperparams: Mapping
    ) -> FloatArray:
        transformed_points = state["pose"].apply(hyperparams["vertices"])
        points_to_pixels = self.get_pixels_points_association(
            transformed_points, hyperparams
        )
        vertex_kernel = self.get_vertex_kernel(state)
        observed_rgbd_per_point = observed_rgbd.at[
            points_to_pixels.x, points_to_pixels.y
        ].get(mode="drop", fill_value=-1.0)
        latent_rgbd_per_point = jnp.concatenate(
            (state["colors"], transformed_points[..., 2]), axis=-1
        )

        scores = jax.vmap(vertex_kernel.logpdf)(
            observed_rgbd_per_point,
            latent_rgbd_per_point,
            1 - state["visibility_prob"],
            state["depth_nonreturn_prob"],
        )
        return scores.sum()

    def get_vertex_kernel(self, state: Mapping) -> PixelRGBDDistribution:
        # Note: The distributions were originally defined for per-pixel computation,
        # but they should work for per-vertex computation as well
        return PixelRGBDDistribution(
            FullPixelColorDistribution(state["color_scale"]),
            FullPixelDepthDistribution(self.near, self.far, state["depth_scale"]),
        )
