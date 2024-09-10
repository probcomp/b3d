import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from genjax.typing import FloatArray, PRNGKey

from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import PixelColorDistribution
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import PixelDepthDistribution


@Pytree.dataclass
class PixelRGBDDistribution(genjax.ExactDensity):
    color_kernel: PixelColorDistribution
    depth_kernel: PixelDepthDistribution

    def sample(
        self, key: PRNGKey, latent_rgbd: FloatArray, *args, **kwargs
    ) -> FloatArray:
        keys = jax.random.split(key, 2)
        observed_color = self.color_kernel.sample(
            keys[0], latent_rgbd[:3], *args, **kwargs
        )
        observed_depth = self.depth_kernel.sample(
            keys[1], latent_rgbd[3], *args, **kwargs
        )
        return jnp.append(observed_color, observed_depth)

    def logpdf(
        self, observed_rgbd: FloatArray, latent_rgbd: FloatArray, *args, **kwargs
    ) -> float:
        color_logpdf = self.color_kernel.logpdf(
            observed_rgbd[:3], latent_rgbd[:3], *args, **kwargs
        )
        depth_logpdf = self.depth_kernel.logpdf(
            observed_rgbd[3], latent_rgbd[3], *args, **kwargs
        )
        return color_logpdf + depth_logpdf
