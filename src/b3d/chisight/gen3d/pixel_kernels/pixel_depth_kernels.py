from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey
from tensorflow_probability.substrates import jax as tfp

from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import is_unexplained
from b3d.modeling_utils import (
    _FIXED_DEPTH_UNIFORM_WINDOW,
    PythonMixtureDistribution,
    truncated_laplace,
)

if TYPE_CHECKING:
    import tensorflow_probability.python.distributions.distribution as dist

DEPTH_NONRETURN_VAL = 0.0
UNEXPLAINED_DEPTH_NONRETURN_PROB = 0.02


@Pytree.dataclass
class PixelDepthDistribution(genjax.ExactDensity):
    """An abstract class that defines the common interface for pixel depth kernels."""

    @abstractmethod
    def sample(self, key: PRNGKey, latent_depth: float, *args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def logpdf(
        self, observed_depth: float, latent_depth: float, *args, **kwargs
    ) -> float:
        raise NotImplementedError


@Pytree.dataclass
class TruncatedLaplacePixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a truncated
    Laplace distribution centered around the latent depth, with the spread
    controlled by depth_scale. The support of the distribution is [near, far].
    """

    near: float = Pytree.static()
    far: float = Pytree.static()
    depth_scale: float
    # the uniform window is used to wrapped the truncated laplace distribution
    # to ensure that the depth generated is within the range of [near, far]
    uniform_window_size: float = Pytree.static(default=_FIXED_DEPTH_UNIFORM_WINDOW)

    def sample(self, key: PRNGKey, latent_depth: float, *args, **kwargs) -> float:
        return truncated_laplace.sample(
            key,
            latent_depth,
            self.depth_scale,
            self.near,
            self.far,
            self.uniform_window_size,
        )

    def logpdf(
        self, observed_depth: float, latent_depth: float, *args, **kwargs
    ) -> float:
        return truncated_laplace.logpdf(
            observed_depth,
            latent_depth,
            self.depth_scale,
            self.near,
            self.far,
            self.uniform_window_size,
        )


@Pytree.dataclass
class UniformPixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from a uniform from
    [near, far]."""

    near: float = Pytree.static()
    far: float = Pytree.static()

    @property
    def _base_dist(self) -> "dist.Distribution":
        return tfp.distributions.Uniform(self.near, self.far)

    def sample(self, key: PRNGKey, *args, **kwargs) -> float:
        return self._base_dist.sample(seed=key)

    def logpdf(self, observed_depth: float, *args, **kwargs) -> float:
        return self._base_dist.log_prob(observed_depth)


@Pytree.dataclass
class DeltaDistribution(genjax.ExactDensity):
    """
    Degenerate discrete distribution (a single point). It assigns probability one
    to the single element in its support.
    """

    value: Any

    def sample(self, key: PRNGKey, *args, **kwargs) -> Any:
        return self.value

    def logpdf(self, sampled_val: Any, *args, **kwargs) -> float:
        return jnp.log(sampled_val == self.value)


@Pytree.dataclass
class MixturePixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from
    mixture(
        [delta(-1), uniform(near, far), laplace(latent_depth; depth_scale)],
        [depth_nonreturn_prob, (1 - depth_nonreturn_prob) * occluded_prob, remaining_prob]
    )

    The support of the distribution is [near, far] ∪ { "nonreturn" }.
    """

    near: float = Pytree.static()
    far: float = Pytree.static()
    depth_scale: float

    @property
    def _nonreturn_dist(self) -> PixelDepthDistribution:
        return DeltaDistribution(DEPTH_NONRETURN_VAL)

    @property
    def _occluded_dist(self) -> PixelDepthDistribution:
        return UniformPixelDepthDistribution(self.near, self.far)

    @property
    def _inlier_dist(self) -> PixelDepthDistribution:
        return TruncatedLaplacePixelDepthDistribution(
            self.near, self.far, self.depth_scale
        )

    @property
    def _mixture_dist(self) -> PythonMixtureDistribution:
        return PythonMixtureDistribution(
            (self._nonreturn_dist, self._occluded_dist, self._inlier_dist)
        )

    def _get_mix_ratio(
        self, occluded_prob: float, depth_nonreturn_prob: float
    ) -> FloatArray:
        return jnp.array(
            (
                depth_nonreturn_prob,
                (1 - depth_nonreturn_prob) * occluded_prob,
                (1 - depth_nonreturn_prob) * (1 - occluded_prob),
            )
        )

    def sample(
        self,
        key: PRNGKey,
        latent_depth: float,
        occluded_prob: float,
        depth_nonreturn_prob: float,
        *args,
        **kwargs,
    ) -> float:
        return self._mixture_dist.sample(
            key,
            self._get_mix_ratio(occluded_prob, depth_nonreturn_prob),
            [(), (), (latent_depth,)],
        )

    def logpdf(
        self,
        observed_depth: float,
        latent_depth: float,
        occluded_prob: float,
        depth_nonreturn_prob: float,
        *args,
        **kwargs,
    ) -> float:
        return self._mixture_dist.logpdf(
            observed_depth,
            self._get_mix_ratio(occluded_prob, depth_nonreturn_prob),
            [(), (), (latent_depth,)],
        )


@Pytree.dataclass
class UnexplainedPixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of a pixel from
    mixture(
        [delta(-1), uniform(near, far)],
        [unexplained_depth_nonreturn_prob, 1 - unexplained_depth_nonreturn_prob]
    ),
    for pixels that are not explained by the latent points.

    The support of the distribution is [near, far] ∪ { "nonreturn" }.
    """

    near: float = Pytree.static()
    far: float = Pytree.static()
    unexplained_depth_nonreturn_prob: float = Pytree.static(
        default=UNEXPLAINED_DEPTH_NONRETURN_PROB
    )

    @property
    def _nonreturn_dist(self) -> PixelDepthDistribution:
        return DeltaDistribution(DEPTH_NONRETURN_VAL)

    @property
    def _uniform_dist(self) -> PixelDepthDistribution:
        return UniformPixelDepthDistribution(self.near, self.far)

    @property
    def _mixture_dist(self) -> PythonMixtureDistribution:
        return PythonMixtureDistribution((self._nonreturn_dist, self._uniform_dist))

    @property
    def _mix_ratio(self) -> FloatArray:
        return jnp.array(
            (
                self.unexplained_depth_nonreturn_prob,
                1 - self.unexplained_depth_nonreturn_prob,
            )
        )

    def sample(
        self,
        key: PRNGKey,
        *args,
        **kwargs,
    ) -> float:
        return self._mixture_dist.sample(key, self._mix_ratio, [(), ()])

    def logpdf(
        self,
        observed_depth: float,
        *args,
        **kwargs,
    ) -> float:
        return self._mixture_dist.logpdf(observed_depth, self._mix_ratio, [(), ()])


@Pytree.dataclass
class FullPixelDepthDistribution(PixelDepthDistribution):
    """A distribution that generates the depth of the pixel according to the
    following rule:

    if no latent point hits the pixel:
        depth ~ mixture(
            [delta(-1), uniform(near, far)],
            [unexplained_depth_nonreturn_prob, 1 - unexplained_depth_nonreturn_prob]
        )
    else:
        mixture(
            [delta(-1), uniform(near, far), laplace(latent_depth; depth_scale)],
            [depth_nonreturn_prob, (1 - depth_nonreturn_prob) * occluded_prob, remaining_prob]
        )
    """

    near: float = Pytree.static()
    far: float = Pytree.static()
    depth_scale: float

    @property
    def _depth_from_latent(self) -> PixelDepthDistribution:
        return MixturePixelDepthDistribution(self.near, self.far, self.depth_scale)

    @property
    def _unexplained_depth(self) -> PixelDepthDistribution:
        return UnexplainedPixelDepthDistribution(self.near, self.far)

    def sample(
        self,
        key: PRNGKey,
        latent_depth: FloatArray,
        occluded_prob: FloatArray,
        depth_nonreturn_prob: float,
        *args,
        **kwargs,
    ) -> FloatArray:
        return jax.lax.cond(
            is_unexplained(latent_depth),
            self._unexplained_depth.sample,  # if no point hits current pixel
            self._depth_from_latent.sample,  # if pixel is being hit by a latent point
            # sample args
            key,
            latent_depth,
            occluded_prob,
            depth_nonreturn_prob,
        )

    def logpdf(
        self,
        observed_depth: FloatArray,
        latent_depth: FloatArray,
        occluded_prob: float,
        depth_nonreturn_prob: float,
        *args,
        **kwargs,
    ) -> FloatArray:
        return jax.lax.cond(
            is_unexplained(latent_depth),
            self._unexplained_depth.logpdf,  # if no point hits current pixel
            self._depth_from_latent.logpdf,  # if pixel is being hit by a latent point
            # logpdf args
            observed_depth,
            latent_depth,
            occluded_prob,
            depth_nonreturn_prob,
        )
