import genjax
import jax
import jax.numpy as jnp
from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import PixelColorDistribution
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
    PixelDepthDistributionForDepthReturn,
)
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey
from jax.random import split


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
class PixelDistFromColorDistAndDepthReturnDist(genjax.ExactDensity):
    """
    Distribution on an RGBD pixel.
    This wraps (1) a color distribution and (2)
    a depth distribution for the case where the depth value is a return.

    This distribution samples a pixel by sampling from the color distribution,
    and either sampling a depth value from the depth distribution or deciding
    that the depth value at this pixel is a non-return.

    Distribution support:
        [0, 1]^3 x ([near, far] + {DEPTH_NONRETURN_VALUE}).

    Args:
    - latent_rgbd
    - color_scale
    - depth_scale
    - depth_nonreturn_prob
    - intrinsics dict containing `"near"` and `"far"` keys.

    [TODO: is there a better name for this class?]
    """

    color_distribution: PixelColorDistribution
    depth_distribution: PixelDepthDistributionForDepthReturn

    def sample(
        self,
        key: PRNGKey,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        depth_nonreturn_prob: float,
        intrinsics: dict,
    ) -> FloatArray:
        k1, k2, k3 = split(key, 3)
        color = self.color_distribution.sample(k1, latent_rgbd[:3], color_scale)
        depth_if_return = self.depth_distribution.sample(
            k2, latent_rgbd[3], depth_scale, intrinsics["near"], intrinsics["far"]
        )
        depth = jnp.where(
            jax.random.bernoulli(k3, depth_nonreturn_prob), 0.0, depth_if_return
        )

        return jnp.concatenate([color, jnp.array([depth])])

    def logpdf(
        self, obs, latent, color_scale, depth_scale, depth_nonreturn_prob, intrinsics
    ):
        color_logpdf = self.color_distribution.logpdf(obs[:3], latent[:3], color_scale)
        depth_logpdf_if_return = self.depth_distribution.logpdf(
            obs[3], latent[3], depth_scale, intrinsics["near"], intrinsics["far"]
        )
        depth_logpdf = jnp.where(
            obs[3] == 0.0,
            jnp.log(depth_nonreturn_prob),
            jnp.log(1 - depth_nonreturn_prob) + depth_logpdf_if_return,
        )
        return color_logpdf + depth_logpdf


@Pytree.dataclass
class PixelRGBDDistribution(genjax.ExactDensity):
    """
    Args:
    - latent_rgbd: 4-array: RGBD value.  (a value of [-1, -1, -1, -1] indicates no point hits here.)
    - color_scale: float
    - depth_scale: float
    - visibility_prob: float
    - depth_nonreturn_prob: float
    - intrinsics: dict
    - depth_nonreturn_prob_for_invisible: float

    The support of the distribution is [0, 1]^3 x ([near, far] + {DEPTH_NONRETURN_VALUE}).

    The generative process is:
    - If latent_rgbd is [-1, -1, -1, -1], the pixel color will be sampled from `outlier_color_distribution`
        and the depth will be sampled from `outlier_depth_distribution`, called with `depth_nonreturn_prob_for_invisible`.
    - Otherwise, with probability `1 - visibility_prob`, the pixel will be sampled in the same way
        as the preceding bullet.
    - Finally, if `latent_rgbd` is not [-1, -1, -1, -1], with probability `visibility_prob`, the pixel
        color will be sampled from `inlier_color_distribution` and the depth will be sampled from `inlier_depth_distribution`
        called with `depth_nonreturn_prob` (rather than `depth_nonreturn_prob_for_invisible`).
    """

    inlier_color_distribution: PixelColorDistribution
    outlier_color_distribution: PixelColorDistribution

    inlier_depth_distribution: PixelDepthDistributionForDepthReturn
    outlier_depth_distribution: PixelDepthDistributionForDepthReturn

    @property
    def inlier_distribution(self):
        return PixelDistFromColorDistAndDepthReturnDist(
            self.inlier_color_distribution, self.inlier_depth_distribution
        )

    @property
    def outlier_distribution(self):
        return PixelDistFromColorDistAndDepthReturnDist(
            self.outlier_color_distribution, self.outlier_depth_distribution
        )

    def sample(
        self,
        key: PRNGKey,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
        intrinsics: dict,
        depth_nonreturn_prob_for_invisible: float,
    ) -> FloatArray:
        k1, k2, k3 = split(key, 3)
        choose_to_be_invisible = jax.random.bernoulli(k1, visibility_prob)
        return jnp.where(
            jnp.logical_or(is_unexplained(latent_rgbd), choose_to_be_invisible),
            self.inlier_distribution.sample(
                k2,
                latent_rgbd,
                color_scale,
                depth_scale,
                depth_nonreturn_prob,
                intrinsics,
            ),
            self.outlier_distribution.sample(
                k3,
                latent_rgbd,
                color_scale,
                depth_scale,
                depth_nonreturn_prob_for_invisible,
                intrinsics,
            ),
        )

    @jax.jit
    def logpdf(
        self,
        observed_rgbd: FloatArray,
        latent_rgbd: FloatArray,
        color_scale: float,
        depth_scale: float,
        visibility_prob: float,
        depth_nonreturn_prob: float,
        intrinsics: dict,
        invisible_depth_nonreturn_prob: float,
    ) -> float:
        log_inlier_prob = self.inlier_distribution.logpdf(
            observed_rgbd,
            latent_rgbd,
            color_scale,
            depth_scale,
            depth_nonreturn_prob,
            intrinsics,
        )
        log_outlier_prob = self.outlier_distribution.logpdf(
            observed_rgbd,
            latent_rgbd,
            color_scale,
            depth_scale,
            invisible_depth_nonreturn_prob,
            intrinsics,
        )
        score_if_latent_is_valid = jnp.logaddexp(
            jnp.log(visibility_prob) + log_inlier_prob,
            jnp.log(1 - visibility_prob) + log_outlier_prob,
        )
        return jnp.where(
            is_unexplained(latent_rgbd), log_outlier_prob, score_if_latent_is_valid
        )
