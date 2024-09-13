from abc import abstractmethod
from typing import TYPE_CHECKING

import genjax
import jax
import jax.numpy as jnp
from b3d.modeling_utils import (
    _FIXED_COLOR_UNIFORM_WINDOW,
    PythonMixtureDistribution,
    renormalized_laplace,
    truncated_laplace,
)
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey
from jax.random import split
from tensorflow_probability.substrates import jax as tfp

if TYPE_CHECKING:
    import tensorflow_probability.python.distributions.distribution as dist

COLOR_MIN_VAL: float = 0.0
COLOR_MAX_VAL: float = 1.0


@Pytree.dataclass
class PixelColorDistribution(genjax.ExactDensity):
    """
    An abstract class that defines the common interface for pixel color kernels.

    Distribuiton args:
    - latent_rgb
    - rgb_scale
    - visibility_prob

    Support:
    - An RGB value in [0, 1]^3.
    """

    @abstractmethod
    def sample(
        self, key: PRNGKey, latent_color: FloatArray, *args, **kwargs
    ) -> FloatArray:
        raise NotImplementedError

    def logpdf(
        self, observed_color: FloatArray, latent_color: FloatArray, *args, **kwargs
    ) -> float:
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
class RenormalizedGaussianPixelColorDistribution(PixelColorDistribution):
    """
    Sample a color from a renormalized Gaussian distribution centered around the given
    latent_color (rgb value), given the color_scale (stddev of the Gaussian).

    The support of the distribution is ([0, 1]^3).
    """

    def sample(self, key, latent_color, color_scale, *args, **kwargs):
        return jax.vmap(
            genjax.truncated_normal.sample, in_axes=(0, 0, None, None, None)
        )(
            split(key, latent_color.shape[0]),
            latent_color,
            color_scale,
            COLOR_MIN_VAL,
            COLOR_MAX_VAL,
        )

    def logpdf_per_channel(
        self, observed_color, latent_color, color_scale, *args, **kwargs
    ):
        return jax.vmap(
            genjax.truncated_normal.logpdf, in_axes=(0, 0, None, None, None)
        )(observed_color, latent_color, color_scale, COLOR_MIN_VAL, COLOR_MAX_VAL)


@Pytree.dataclass
class RenormalizedLaplacePixelColorDistribution(PixelColorDistribution):
    """
    Sample a color from a renormalized Laplace distribution centered around the given
    latent_color (rgb value), given the color_scale (scale of the laplace).

    The support of the distribution is ([0, 1]^3).
    """

    def sample(self, key, latent_color, color_scale, *args, **kwargs):
        return jax.vmap(renormalized_laplace.sample, in_axes=(0, 0, None, None, None))(
            split(key, latent_color.shape[0]),
            latent_color,
            color_scale,
            COLOR_MIN_VAL,
            COLOR_MAX_VAL,
        )

    def logpdf_per_channel(
        self, observed_color, latent_color, color_scale, *args, **kwargs
    ):
        return jax.vmap(renormalized_laplace.logpdf, in_axes=(0, 0, None, None, None))(
            observed_color, latent_color, color_scale, COLOR_MIN_VAL, COLOR_MAX_VAL
        )


@Pytree.dataclass
class TruncatedLaplacePixelColorDistribution(PixelColorDistribution):
    """A distribution that generates the color of a pixel from a truncated
    Laplace distribution centered around the latent color, with the spread
    controlled by color_scale. The support of the distribution is ([0, 1]^3).
    """

    # the uniform window is used to wrapped the truncated laplace distribution
    # to ensure that the color generated is within the range of [0, 1]
    uniform_window_size: float = Pytree.static(default=_FIXED_COLOR_UNIFORM_WINDOW)

    def sample(
        self,
        key: PRNGKey,
        latent_color: FloatArray,
        color_scale: FloatArray,
        *args,
        **kwargs,
    ) -> FloatArray:
        return jax.vmap(
            lambda k, color: truncated_laplace.sample(
                k,
                color,
                color_scale,
                COLOR_MIN_VAL,
                COLOR_MAX_VAL,
                self.uniform_window_size,
            ),
            in_axes=(0, 0),
        )(jax.random.split(key, latent_color.shape[0]), latent_color)

    def logpdf_per_channel(
        self,
        observed_color: FloatArray,
        latent_color: FloatArray,
        color_scale: FloatArray,
        *args,
        **kwargs,
    ) -> FloatArray:
        return jax.vmap(
            lambda obs, latent: truncated_laplace.logpdf(
                obs,
                latent,
                color_scale,
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
    branch) and a uniform distribution (occluded branch). The mixture is
    controlled by the occluded_prob parameter. The support of the
    distribution is ([0, 1]^3).
    """

    @property
    def _occluded_dist(self) -> PixelColorDistribution:
        return UniformPixelColorDistribution()

    @property
    def _inlier_dist(self) -> PixelColorDistribution:
        return TruncatedLaplacePixelColorDistribution()

    @property
    def _mixture_dists(self) -> tuple[PixelColorDistribution, PixelColorDistribution]:
        return (self._occluded_dist, self._inlier_dist)

    def _get_mix_ratio(self, visibility_prob: float) -> FloatArray:
        return jnp.array((1 - visibility_prob, visibility_prob))

    def sample(
        self,
        key: PRNGKey,
        latent_color: FloatArray,
        color_scale: FloatArray,
        visibility_prob: float,
        *args,
        **kwargs,
    ) -> FloatArray:
        return PythonMixtureDistribution(self._mixture_dists).sample(
            key, self._get_mix_ratio(visibility_prob), [(), (latent_color, color_scale)]
        )

    def logpdf_per_channel(
        self,
        observed_color: FloatArray,
        latent_color: FloatArray,
        color_scale: FloatArray,
        visibility_prob: float,
        *args,
        **kwargs,
    ) -> FloatArray:
        # Since the mixture model class does not keep the per-channel information,
        # we have to redefine this method to allow for testing
        logprobs = []
        for dist, prob in zip(
            self._mixture_dists, self._get_mix_ratio(visibility_prob)
        ):
            logprobs.append(
                dist.logpdf_per_channel(observed_color, latent_color, color_scale)
                + jnp.log(prob)
            )

        return jnp.logaddexp(*logprobs)
