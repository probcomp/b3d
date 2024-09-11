from abc import abstractmethod

import genjax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey

from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import PixelColorDistribution
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import PixelDepthDistribution


def is_unexplained(latent_value: FloatArray) -> bool:
    """
    Check if a given `latent_value` value given to a pixel
    indicates that no latent point hits a pixel.
    This is done by checking if any of the latent color values
    are negative.

    Args:
        latent_value (FloatArray): The latent color of the pixel.

    Returns:
        bool: True is none of the latent point hits the pixel, False otherwise.
    """
    return jnp.any(latent_value < 0.0)


@Pytree.dataclass
class PixelRGBDDistribution(genjax.ExactDensity):
    """
    Distribution args:
    - latent_rgbd: 4-array: RGBD value.  (a value of [-1, -1, -1, -1] indicates no point hits here.)
    - color_scale: float
    - depth_scale: float
    - visibility_prob: float
    - depth_nonreturn_prob: float

    The support of the distribution is [0, 1]^3 x ([near, far] + {DEPTH_NONRETURN_VALUE}).

    Note that this distribution expects the observed_rgbd to be valid. If an invalid
    pixel is observed, the logpdf will return -inf.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKey,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
    ) -> FloatArray:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self,
        observed_rgbd: FloatArray,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
    ) -> float:
        raise NotImplementedError


@Pytree.dataclass
class FullPixelRGBDDistribution(PixelRGBDDistribution):
    """
    Args:
    - latent_rgbd: 4-array: RGBD value.  (a value of [-1, -1, -1, -1] indicates no point hits here.)
    - color_scale: float
    - depth_scale: float
    - visibility_prob: float

    The support of the distribution is [0, 1]^3 x ([near, far] + {DEPTH_NONRETURN_VALUE}).
    """

    inlier_color_distribution: PixelColorDistribution
    outlier_color_distribution: PixelColorDistribution

    inlier_depth_distribution: PixelDepthDistribution
    outlier_depth_distribution: PixelDepthDistribution

    def sample(
        self,
        key: PRNGKey,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
    ) -> FloatArray:
        # TODO: Implement this
        return jnp.ones((4,)) * 0.5

    def logpdf(
        self,
        observed_rgbd: FloatArray,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
    ) -> float:
        total_log_prob = 0.0

        is_depth_non_return = observed_rgbd[3] == 0.0

        # Is visible
        total_visible_log_prob = 0.0
        # color term
        total_visible_log_prob += self.inlier_color_distribution.logpdf(
            observed_rgbd[:3], latent_rgbd[:3], color_scale
        )
        # depth term
        total_visible_log_prob += jnp.where(
            is_depth_non_return,
            jnp.log(depth_nonreturn_prob),
            jnp.log(1 - depth_nonreturn_prob)
            + self.inlier_depth_distribution.logpdf(
                observed_rgbd[3], latent_rgbd[3], depth_scale
            ),
        )

        # Is not visible
        total_not_visible_log_prob = 0.0
        # color term
        outlier_color_log_prob = self.outlier_color_distribution.logpdf(
            observed_rgbd[:3], latent_rgbd[:3], color_scale
        )
        outlier_depth_log_prob = self.outlier_depth_distribution.logpdf(
            observed_rgbd[3], latent_rgbd[3], depth_scale
        )

        total_not_visible_log_prob += outlier_color_log_prob
        # depth term
        total_not_visible_log_prob += jnp.where(
            is_depth_non_return,
            jnp.log(depth_nonreturn_prob),
            jnp.log(1 - depth_nonreturn_prob) + outlier_depth_log_prob,
        )

        total_log_prob += jnp.logaddexp(
            jnp.log(visibility_prob) + total_visible_log_prob,
            jnp.log(1 - visibility_prob) + total_not_visible_log_prob,
        )
        return jnp.where(
            jnp.any(is_unexplained(latent_rgbd)),
            outlier_color_log_prob + outlier_depth_log_prob,
            total_log_prob,
        )
