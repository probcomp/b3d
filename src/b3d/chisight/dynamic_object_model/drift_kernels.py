from abc import abstractmethod
from typing import Sequence

import genjax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import ArrayLike, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from b3d.chisight.dense.likelihoods.other_likelihoods import PythonMixturePixelModel
from b3d.chisight.dynamic_object_model.likelihoods.kfold_image_kernel import (
    _FIXED_COLOR_UNIFORM_WINDOW,
    truncated_color_laplace,
    truncated_laplace,
)


@Pytree.dataclass
class DriftKernel(genjax.ExactDensity):
    """An abstract class that defines the common interface for drift kernels."""

    @abstractmethod
    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        raise NotImplementedError


@Pytree.dataclass
class UniformDriftKernel(DriftKernel):
    """A drift kernel that samples a new value from a uniform distribution centered
    around the previous value. The range of the uniform distribution may shrink
    to ensure that the new value is within the bounds of [min_val, max_val].

    Support: [max(min_val, prev_value - max_shift), min(max_val, prev_value + max_shift)]
    """

    max_shift: float = Pytree.static()
    min_val: float = Pytree.static()
    max_val: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return self._base_dist(prev_value).sample(seed=key)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return self._base_dist(prev_value).log_prob(new_value)

    def _base_dist(self, prev_value: ArrayLike):
        """Returns a uniform distribution centered around prev_value, bounded by
        min_val and max_val."""
        low = jnp.maximum(prev_value - self.max_shift, self.min_val)
        high = jnp.minimum(prev_value + self.max_shift, self.max_val)
        return tfp.distributions.Uniform(low, high)


@Pytree.dataclass
class UniformColorDriftKernel(UniformDriftKernel):
    """A specialized uniform drift kernel with fixed min_val and max_val, with
    additional logics to handle the color channels jointly.

    Support: [max(0.0, prev_value - max_shift), min(1.0, prev_value + max_shift)]
    """

    max_shift: float = Pytree.static()
    min_val: float = Pytree.static(default=0.0, init=False)
    max_val: float = Pytree.static(default=1.0, init=False)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        # the summation at the end is to ensure that we get a single value for
        # the 3 channels (instead of 3 separate values)
        return super().logpdf(new_value, prev_value).sum()


@Pytree.dataclass
class LaplaceDriftKernel(DriftKernel):
    """A drift kernel that samples from a truncated Laplace distribution centered
    at the previous value. Values outside of the bounds will be resampled from a
    small uniform window at the boundary. This is a thin wrapper around the
    truncated_laplace distribution to provide a consistent interface with other
    drift kernels.

    Support: [min_val, max_val]
    """

    scale: float = Pytree.static()
    min_val: float = Pytree.static()
    max_val: float = Pytree.static()
    uniform_window_size: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return truncated_laplace.sample(
            key, prev_value, self.scale, self.uniform_window_size
        )

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return truncated_laplace.logpdf(
            new_value, prev_value, self.scale, self.uniform_window_size
        )


@Pytree.dataclass
class LaplaceColorDriftKernel(DriftKernel):
    """A drift kernel that samples the 3 channels of the color from a specialized
    truncated Laplace distribution, centered at the previous color. Values outside
    of the bounds will be resampled from a small uniform window at the boundary.
    This is a thin wrapper around the truncated_color_laplace distribution to
    provide a consistent interface with other drift kernels.

    Support: [0.0, 1.0]
    """

    scale: float = Pytree.static()
    uniform_window_size: float = Pytree.static(default=_FIXED_COLOR_UNIFORM_WINDOW)

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return truncated_color_laplace.sample(
            key, prev_value, self.scale, self.uniform_window_size
        )

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return truncated_color_laplace.logpdf(
            new_value, prev_value, self.scale, self.uniform_window_size
        )


@Pytree.dataclass
class GaussianDriftKernel(DriftKernel):
    """A drift kernel that samples from a truncated Gaussian distribution centered
    at the previous value. Values outside of the bounds will be renormalized.

    Support: [min_val, max_val]
    """

    scale: float = Pytree.static()
    min_val: float = Pytree.static()
    max_val: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return self._base_dist(prev_value).sample(seed=key)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return self._base_dist(prev_value).log_prob(new_value)

    def _base_dist(self, prev_value: ArrayLike):
        return tfp.distributions.TruncatedNormal(
            loc=prev_value, scale=self.scale, low=self.min_val, high=self.max_val
        )


@Pytree.dataclass
class GaussianColorDriftKernel(GaussianDriftKernel):
    """A specialized Gaussian drift kernel that samples from a truncated Gaussian
    distribution centered at the previous value. Values outside of the bounds
    will be renormalized.

    Support: [0.0, 1.0]
    """

    scale: float = Pytree.static()
    min_val: float = Pytree.static(default=0.0, init=False)
    max_val: float = Pytree.static(default=1.0, init=False)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        # the summation at the end is to ensure that we get a single value for
        # the 3 channels (instead of 3 separate values)
        return super().logpdf(new_value, prev_value).sum()


@Pytree.dataclass
class MixtureDriftKernel(DriftKernel):
    """A drift kernel that samples from a mixture of distributions according to
    the probabilities specified in the `mix_ratio`.
    """

    dists: Sequence[DriftKernel] = Pytree.static()
    mix_ratio: ArrayLike = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return PythonMixturePixelModel(self.dists).sample(
            key, self.mix_ratio, [(prev_value,)] * len(self.dists)
        )

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return PythonMixturePixelModel(self.dists).logpdf(
            new_value,
            self.mix_ratio,
            [(prev_value,)] * len(self.dists),
        )
