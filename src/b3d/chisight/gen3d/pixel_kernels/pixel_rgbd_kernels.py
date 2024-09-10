import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey

from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import PixelColorDistribution
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import PixelDepthDistribution


@Pytree.dataclass
class PixelRGBDDistribution(genjax.ExactDensity):
    """
    Distribution args:
    - latent_rgbd: 4-array: RGBD value.  (Should be [-1, -1, -1, -1] to indicate no point hits here.)
    - rgb_scale: float
    - depth_scale: float
    - visibility_prob: float
    - depth_nonreturn_prob: float

    The support of the distribution is [0, 1]^3 x ([near, far] + {DEPTH_NONRETURN_VALUE}).
    """

    color_kernel: PixelColorDistribution
    depth_kernel: PixelDepthDistribution

    def sample(
        self,
        key: PRNGKey,
        latent_rgbd: FloatArray,
        rgb_scale,
        depth_scale,
        visibility_prob,
        depth_nonreturn_prob,
    ) -> FloatArray:
        keys = jax.random.split(key, 2)
        observed_color = self.color_kernel.sample(
            keys[0], latent_rgbd[:3], rgb_scale, visibility_prob
        )
        observed_depth = self.depth_kernel.sample(
            keys[1], latent_rgbd[3], depth_scale, visibility_prob, depth_nonreturn_prob
        )
        return jnp.append(observed_color, observed_depth)

    def logpdf(
        self,
        observed_rgbd: FloatArray,
        latent_rgbd: FloatArray,
        rgb_scale,
        depth_scale,
        visibility_prob,
        depth_nonreturn_prob,
    ) -> float:
        color_logpdf = self.color_kernel.logpdf(
            observed_rgbd[:3], latent_rgbd[:3], rgb_scale, visibility_prob
        )
        depth_logpdf = self.depth_kernel.logpdf(
            observed_rgbd[3],
            latent_rgbd[3],
            depth_scale,
            visibility_prob,
            depth_nonreturn_prob,
        )
        return color_logpdf + depth_logpdf
