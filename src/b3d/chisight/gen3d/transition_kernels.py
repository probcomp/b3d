from abc import abstractmethod
from typing import Sequence

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import ArrayLike, PRNGKey
from tensorflow_probability.substrates import jax as tfp

import b3d.chisight.gen3d.uniform_distributions as uf
from b3d import Pose
from b3d.chisight.dense.likelihoods.other_likelihoods import PythonMixturePixelModel
from b3d.modeling_utils import (
    renormalized_color_laplace,
    truncated_color_laplace,
    truncated_laplace,
)

_FIXED_COLOR_UNIFORM_WINDOW = 1 / 255


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
class CenteredUniformColorDriftKernel(DriftKernel):
    """A specialized uniform drift kernel with fixed min_val and max_val, with
    additional logics to handle the color channels jointly.

    Support: [max(0.0, prev_value - max_shift), min(1.0, prev_value + max_shift)]
    """

    epsilon: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return jax.vmap(
            uf.NiceTruncatedCenteredUniform(self.epsilon, 0.0, 1.0).sample,
            in_axes=(0, 0),
        )(jax.split(key, 3), prev_value)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return jax.vmap(
            uf.NiceTruncatedCenteredUniform(self.epsilon, 0.0, 1.0).logpdf,
            in_axes=(0, 0),
        )(new_value, prev_value).sum()


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
class RenormalizedLaplaceColorDriftKernel(DriftKernel):
    scale: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        return renormalized_color_laplace.sample(key, prev_value, self.scale)

    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        return renormalized_color_laplace.logpdf(new_value, prev_value, self.scale)


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
    """
    [TODO: check that the math for the orientation is correct.]
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


@Pytree.dataclass
class GaussianVMFPoseDriftKernel(DriftKernel):
    std: float = Pytree.static()
    concentration: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose):
        return Pose.sample_gaussian_vmf_pose(
            key, prev_pose, self.std, self.concentration
        )

    def logpdf(self, new_pose, prev_pose) -> ArrayLike:
        return Pose.logpdf_gaussian_vmf_pose(
            new_pose, prev_pose, self.std, self.concentration
        )


# Discrete Kernels


@Pytree.dataclass
class DiscreteFlipKernel(genjax.ExactDensity):
    """
    When `support` has 1 element, this is the delta distribution at that value.
    When `support` has more than 1 element, this is the distribution which,
    given `prev_value`, returns `prev_value` with probability `1 - p_change_to_different_value`,
    and with probability `p_change_to_different_value`, returns a value from `support`
    other than `prev_value`.
    """

    p_change_to_different_value: float = Pytree.static()
    support: ArrayLike = Pytree.static()

    def sample(self, key: PRNGKey, prev_value):
        if len(self.support) == 1:
            return self.support[0]
        else:
            should_resample = jax.random.bernoulli(
                key, self.p_change_to_different_value
            )
            idx_of_prev_value = jnp.argmin(self.support == prev_value)
            idx = jax.random.randint(key, (), 0, len(self.support) - 1)
            idx = jnp.where(idx >= idx_of_prev_value, idx + 1, idx)
            return jnp.where(should_resample, self.support[idx], prev_value)

    def logpdf(self, new_value, prev_value):
        # Write code to compute the logpdf of this flipping kernel.
        match = new_value == prev_value
        number_of_other_values = len(self.support) - 1

        if len(self.support) > 1:
            return jnp.where(
                match,
                jnp.log(1 - self.p_change_to_different_value),
                jnp.log(self.p_change_to_different_value * 1 / number_of_other_values),
            )
        else:
            return jnp.log(self.p_change_to_different_value * 0.0 + 1.0)
