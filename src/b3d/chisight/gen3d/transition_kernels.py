from abc import abstractmethod
from typing import Sequence

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import ArrayLike, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from b3d import Pose
from b3d.chisight.dense.likelihoods.other_likelihoods import PythonMixturePixelModel


@Pytree.dataclass
class TruncatedLaplace(genjax.ExactDensity):
    """
    This is a distribution on the interval (low, high).
    The generative process is:
    1. Sample x ~ laplace(loc, scale).
    2. If x < low, sample y ~ uniform(low, low + uniform_window_size) and return y.
    3. If x > high, sample y ~ uniform(high - uniform_window_size, high) and return y.
    4. Otherwise, return x.

    Args:
    - loc: float
    - scale: float
    - low: float
    - high: float
    - uniform_window_size: float

    Support:
    - x in (low, high) [a float]
    """

    def sample(self, key, loc, scale, low, high, uniform_window_size):
        assert low < high
        assert low + uniform_window_size < high - uniform_window_size
        k1, k2 = jax.random.split(key, 2)
        x = tfp.distributions.Laplace(loc, scale).sample(seed=k1)
        u = jax.random.uniform(k2, ()) * uniform_window_size
        return jnp.where(
            x > high, high - uniform_window_size + u, jnp.where(x < low, low + u, x)
        )

    def logpdf(self, obs, loc, scale, low, high, uniform_window_size):
        assert low < high
        assert low + uniform_window_size < high - uniform_window_size
        laplace_logpdf = tfp.distributions.Laplace(loc, scale).log_prob(obs)
        laplace_logp_below_low = tfp.distributions.Laplace(loc, scale).log_cdf(low)
        laplace_logp_above_high = tfp.distributions.Laplace(
            loc, scale
        ).log_survival_function(high)
        log_window_size = jnp.log(uniform_window_size)

        return jnp.where(
            jnp.logical_and(
                low + uniform_window_size < obs, obs < high - uniform_window_size
            ),
            laplace_logpdf,
            jnp.where(
                obs < low + uniform_window_size,
                jnp.logaddexp(laplace_logp_below_low - log_window_size, laplace_logpdf),
                jnp.logaddexp(
                    laplace_logp_above_high - log_window_size, laplace_logpdf
                ),
            ),
        )


truncated_laplace = TruncatedLaplace()


_FIXED_COLOR_UNIFORM_WINDOW = 1 / 255
_FIXED_DEPTH_UNIFORM_WINDOW = 0.01


@Pytree.dataclass
class TruncatedColorLaplace(genjax.ExactDensity):
    """
    Args:
    - loc: (3,) array (loc for R, G, B channels)
    - shared_scale: float (scale, shared across R, G, B channels)
    - uniform_window_size: float [optional; defaults to 1/255]

    Support:
    - rgb in [0, 1]^3 [a 3D array]
    """

    def sample(
        self, key, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda k, lc: truncated_laplace.sample(
                k, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(jax.random.split(key, loc.shape[0]), loc)

    def logpdf(
        self, obs, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda o, lc: truncated_laplace.logpdf(
                o, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(obs, loc).sum()


truncated_color_laplace = TruncatedColorLaplace()


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
        return self._base_dist(prev_value).log_prob(new_value).sum()

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
class LaplaceNotTruncatedDriftKernel(DriftKernel):
    """A drift kernel that samples the 3 channels of the color from a specialized
    truncated Laplace distribution, centered at the previous color. Values outside
    of the bounds will be resampled from a small uniform window at the boundary.
    This is a thin wrapper around the truncated_color_laplace distribution to
    provide a consistent interface with other drift kernels.

    Support: [0.0, 1.0]
    """

    scale: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return genjax.laplace.sample(key, prev_value, self.scale)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return jax.scipy.stats.laplace.logpdf(new_value, prev_value, self.scale).sum()


@Pytree.dataclass
class LaplaceColorDriftKernel(DriftKernel):
    """A drift kernel that samples the 3 channels of the color from a specialized
    truncated Laplace distribution, centered at the previous color. Values outside
    of the bounds will be resampled from a small uniform window at the boundary.
    This is a thin wrapper around the truncated_color_laplace distribution to
    provide a consistent interface with other drift kernels.

    Support: [0.0, 1.0]^3
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
class LaplaceNotTruncatedColorDriftKernel(DriftKernel):
    """A drift kernel that samples the 3 channels of the color from a specialized
    truncated Laplace distribution, centered at the previous color. Values may
    go outside of the valid color range ([0.0, 1.0]^3).

    Support: [-inf, inf]^3
    """

    scale: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return genjax.laplace.sample(key, prev_value, self.scale)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return jax.scipy.stats.laplace.logpdf(new_value, prev_value, self.scale).sum()


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


# Pose Drift Kernels


@Pytree.dataclass
class UniformPoseDriftKernel(DriftKernel):
    """A specialized uniform drift kernel with fixed min_val and max_val, with
    additional logics to handle the color channels jointly.

    Support: [max(0.0, prev_value - max_shift), min(1.0, prev_value + max_shift)]
    """

    max_shift: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose):
        keys = jax.random.split(key, 2)
        pos = (
            jax.random.uniform(keys[0], (3,)) * (2 * self.max_shift)
            - self.max_shift
            + prev_pose.position
        )
        quat = jax.random.normal(keys[1], (4,))
        quat = quat / jnp.linalg.norm(quat)
        return Pose(pos, quat)

    def logpdf(self, new_pose, prev_pose) -> ArrayLike:
        position_delta = new_pose.pos - prev_pose.pos
        valid = jnp.all(jnp.abs(position_delta) < self.max_shift)
        position_score = jnp.log(
            (valid * 1.0) * (jnp.ones_like(position_delta) / (2 * self.max_shift))
        ).sum()
        return position_score + jnp.pi**2


# Discrete Kernels

# TODO: add back in the base class for discretekernel.
# I removed it since its listed interface had become
# out of sync with `DiscreteFlipKernel`.


@Pytree.dataclass
class DiscreteFlipKernel(genjax.ExactDensity):
    resample_probability: float = Pytree.static()
    support: ArrayLike = Pytree.static()

    def sample(self, key: PRNGKey, prev_value):
        should_resample = jax.random.bernoulli(key, self.resample_probability)
        return (
            should_resample
            * self.support.at[jax.random.choice(key, len(self.support))].get()
            + (1 - should_resample) * prev_value
        )

    def logpdf(self, new_value, prev_value):
        # Write code to compute the logpdf of this flipping kernel.

        resample_probability = self.resample_probability
        support = self.support

        match = new_value == prev_value
        number_of_other_values = len(support) - 1

        log_probability_of_non_matched_values = jnp.where(
            number_of_other_values > 0.0,
            jnp.log(resample_probability) - jnp.log(number_of_other_values),
            jnp.log(0.0),
        )
        log_total_probability_of_non_matched_values = (
            log_probability_of_non_matched_values + jnp.log(len(support) - 1)
        )
        log_probability_of_match = jnp.log(
            1.0 - jnp.exp(log_total_probability_of_non_matched_values)
        )
        logprob = jnp.logaddexp(
            jnp.log(match) + log_probability_of_match,
            log_probability_of_non_matched_values + jnp.log(1.0 - match),
        )
        return logprob
