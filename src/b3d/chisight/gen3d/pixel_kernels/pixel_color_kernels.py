from abc import abstractmethod
from typing import TYPE_CHECKING

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey, ScalarBool, ScalarFloat
from tensorflow_probability.substrates import jax as tfp

from b3d.chisight.dense.likelihoods.other_likelihoods import PythonMixturePixelModel
from b3d.chisight.dynamic_object_model.likelihoods.kfold_image_kernel import (
    _FIXED_COLOR_UNIFORM_WINDOW,
    truncated_laplace,
)

if TYPE_CHECKING:
    import tensorflow_probability.python.distributions.distribution as dist

COLOR_MIN_VAL: float = 0.0
COLOR_MAX_VAL: float = 1.0


def is_unexplained(latent_color: FloatArray) -> ScalarBool:
    """A heuristic to check if a pixel does not have any latent point that hits
    it.

    Args:
        latent_color (FloatArray): The latent color of the pixel.

    Returns:
        bool: True is none of the latent point hits the pixel, False otherwise.
    """
    return jnp.any(latent_color < 0.0)


@Pytree.dataclass
class PixelColorDistribution(genjax.ExactDensity):
    """An abstract class that defines the common interface for pixel color kernels."""

    @abstractmethod
    def sample(
        self, key: PRNGKey, latent_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        raise NotImplementedError

    def logpdf(
        self, observed_color: FloatArray, latent_color: FloatArray, *args, **kwargs
    ) -> ScalarFloat:
        return self.logpdf_per_channel(
            observed_color, latent_color, *args, **kwargs
        ).sum()

    @abstractmethod
    def logpdf_per_channel(
        self, observed_color: FloatArray, latent_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        """Return an array of logpdf values, one for each channel. This is useful
        for testing purposes."""
        raise NotImplementedError


@Pytree.dataclass
class TruncatedLaplacePixelColorDistribution(PixelColorDistribution):
    """A distribution that generates the color of a pixel from a truncated
    Laplace distribution centered around the latent color, with the spread
    controlled by color_scale. The support of the distribution is ([0, 1]^3).
    """

    color_scale: ScalarFloat
    # the uniform window is used to wrapped the truncated laplace distribution
    # to ensure that the color generated is within the range of [0, 1]
    uniform_window_size: ScalarFloat = Pytree.static(
        default=_FIXED_COLOR_UNIFORM_WINDOW
    )

    def sample(
        self, key: PRNGKey, latent_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        return jax.vmap(
            lambda k, color: truncated_laplace.sample(
                k,
                color,
                self.color_scale,
                COLOR_MIN_VAL,
                COLOR_MAX_VAL,
                self.uniform_window_size,
            ),
            in_axes=(0, 0),
        )(jax.random.split(key, latent_color.shape[0]), latent_color)

    def logpdf_per_channel(
        self, observed_color: FloatArray, latent_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        return jax.vmap(
            lambda obs, latent: truncated_laplace.logpdf(
                obs,
                latent,
                self.color_scale,
                COLOR_MIN_VAL,
                COLOR_MAX_VAL,
                self.uniform_window_size,
            ),
            in_axes=(0, 0),
        )(observed_color, latent_color)


@Pytree.dataclass
class UniformPixelColorDistribution(PixelColorDistribution):
    """A distribution that generates the color of a pixel from a uniform on the
    RGB space ([0, 1]^3).
    """

    @property
    def _base_dist(self) -> "dist.Distribution":
        return tfp.distributions.Uniform(COLOR_MIN_VAL, COLOR_MAX_VAL)

    def sample(self, key: PRNGKey, *args, **kwargs) -> FloatArray:
        return self._base_dist.sample(seed=key, sample_shape=(3,))

    def logpdf_per_channel(
        self, observed_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        return self._base_dist.log_prob(observed_color)


@Pytree.dataclass
class MixturePixelColorDistribution(PixelColorDistribution):
    """A distribution that generates the color of a pixel from a mixture of a
    truncated Laplace distribution centered around the latent color (inlier
    branch) and a uniform distribution (outlier branch). The mixture is
    controlled by the color_outlier_prob parameter. The support of the
    distribution is ([0, 1]^3).
    """

    color_scale: ScalarFloat

    @property
    def _inlier_dist(self) -> PixelColorDistribution:
        return TruncatedLaplacePixelColorDistribution(self.color_scale)

    @property
    def _outlier_dist(self) -> PixelColorDistribution:
        return UniformPixelColorDistribution()

    @property
    def _mixture_dists(self) -> tuple[PixelColorDistribution, PixelColorDistribution]:
        return (self._inlier_dist, self._outlier_dist)

    def get_mix_ratio(self, color_outlier_prob: ScalarFloat) -> FloatArray:
        return jnp.array((1 - color_outlier_prob, color_outlier_prob))

    def sample(
        self,
        key: PRNGKey,
        latent_color: FloatArray,
        color_outlier_prob: ScalarFloat,
        *args,
        **kwargs,
    ) -> FloatArray:
        return PythonMixturePixelModel(self._mixture_dists).sample(
            key, self.get_mix_ratio(color_outlier_prob), [(latent_color,), ()]
        )

    def logpdf_per_channel(
        self,
        observed_color: FloatArray,
        latent_color: FloatArray,
        color_outlier_prob: ScalarFloat,
        *args,
        **kwargs,
    ) -> FloatArray:
        # Since the mixture model class does not keep the per-channel information,
        # we have to redefine this method to allow for testing
        logprobs = []
        for dist, prob in zip(
            self._mixture_dists, self.get_mix_ratio(color_outlier_prob)
        ):
            logprobs.append(
                dist.logpdf_per_channel(observed_color, latent_color) + jnp.log(prob)
            )

        return jnp.logaddexp(*logprobs)


@Pytree.dataclass
class FullPixelColorDistribution(PixelColorDistribution):
    """A distribution that generates the color of the pixel according to the
    following rule:

    if no latent point hits the pixel:
        color ~ uniform(0, 1)
    else:
        color ~ mixture(
            [truncated_laplace(latent_color; color_scale), uniform(0, 1)],
            [1 - color_outlier_prob, color_outlier_prob]
        )
    """

    color_scale: ScalarFloat

    @property
    def _color_from_latent(self) -> PixelColorDistribution:
        return MixturePixelColorDistribution(self.color_scale)

    @property
    def _unexplained_color(self) -> PixelColorDistribution:
        return UniformPixelColorDistribution()

    def sample(
        self,
        key: PRNGKey,
        latent_color: FloatArray,
        color_outlier_prob: FloatArray,
        *args,
        **kwargs,
    ) -> FloatArray:
        # Check if any of the latent point hits the current pixel
        is_explained = ~is_unexplained(latent_color)

        return jax.lax.cond(
            is_explained,
            self._color_from_latent.sample,  # if pixel is being hit by a latent point
            self._unexplained_color.sample,  # if no point hits current pixel
            # sample args
            key,
            latent_color,
            color_outlier_prob,
        )

    def logpdf_per_channel(
        self,
        observed_color: FloatArray,
        latent_color: FloatArray,
        color_outlier_prob: ScalarFloat,
        *args,
        **kwargs,
    ) -> FloatArray:
        # Check if any of the latent point hits the current pixel
        is_explained = ~is_unexplained(latent_color)

        return jax.lax.cond(
            is_explained,
            self._color_from_latent.logpdf_per_channel,  # if pixel is being hit by a latent point
            self._unexplained_color.logpdf_per_channel,  # if no point hits current pixel
            # logpdf args
            observed_color,
            latent_color,
            color_outlier_prob,
        )
