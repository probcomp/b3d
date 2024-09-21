from abc import abstractmethod
from typing import TYPE_CHECKING

import genjax
from b3d.modeling_utils import (
    _FIXED_DEPTH_UNIFORM_WINDOW,
    renormalized_laplace,
    truncated_laplace,
)
from genjax import Pytree
from genjax.typing import PRNGKey
from tensorflow_probability.substrates import jax as tfp

if TYPE_CHECKING:
    import tensorflow_probability.python.distributions.distribution as dist

DEPTH_NONRETURN_VAL = 0.0
UNEXPLAINED_DEPTH_NONRETURN_PROB = 0.02


@Pytree.dataclass
class PixelDepthDistribution(genjax.ExactDensity):
    """
    An abstract class that defines the common interface for pixel depth kernels.

    Distribution args:
    - latent_depth
    - depth_scale
    - near
    - far

    Support: depth value in [near, far], or DEPTH_NONRETURN_VAL.
    """

    @abstractmethod
    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        raise NotImplementedError


@Pytree.dataclass
class RenormalizedGaussianPixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a Gaussian
    distribution centered around the latent depth, with the spread controlled
    by depth_scale. The support of the distribution is [near, far].
    """

    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return genjax.truncated_normal.sample(key, latent_depth, depth_scale, near, far)

    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return genjax.truncated_normal.logpdf(
            observed_depth, latent_depth, depth_scale, near, far
        )


@Pytree.dataclass
class RenormalizedLaplacePixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a Laplace
    distribution centered around the latent depth, with the spread controlled
    by depth_scale. The support of the distribution is [near, far].
    """

    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return renormalized_laplace.sample(key, latent_depth, depth_scale, near, far)

    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return renormalized_laplace.logpdf(
            observed_depth, latent_depth, depth_scale, near, far
        )


@Pytree.dataclass
class TruncatedLaplacePixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a truncated
    Laplace distribution centered around the latent depth, with the spread
    controlled by depth_scale. The support of the distribution is [near, far].
    """

    # the uniform window is used to wrapped the truncated laplace distribution
    # to ensure that the depth generated is within the range of [near, far]
    uniform_window_size: float = Pytree.static(default=_FIXED_DEPTH_UNIFORM_WINDOW)

    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return truncated_laplace.sample(
            key,
            latent_depth,
            depth_scale,
            near,
            far,
            self.uniform_window_size,
        )

    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return truncated_laplace.logpdf(
            observed_depth,
            latent_depth,
            depth_scale,
            near,
            far,
            self.uniform_window_size,
        )


@Pytree.dataclass
class UniformPixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a uniform from
    [near, far]."""

    def _base_dist(self, near, far) -> "dist.Distribution":
        return tfp.distributions.Uniform(near, far)

    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return self._base_dist(near, far).sample(seed=key)

    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        depth_scale: float,
        near: float,
        far: float,
        *args,
        **kwargs,
    ) -> float:
        return self._base_dist(near, far).log_prob(observed_depth)
