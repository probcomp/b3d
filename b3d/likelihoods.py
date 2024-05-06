import genjax
import b3d
import jax.numpy as jnp

import jax
from jax import jit
from functools import partial

from tensorflow_probability.substrates import jax as tfp

# This is from ChatGPT and I'm not totally sure it's right.
# This test passes:
# for i in range(100):
#     subkey, key = jax.random.split(key)
#     rgb = jax.random.uniform(subkey, (3,))
#     lab = rgb_to_lab(rgb)
#     rgb2 = lab_to_rgb(lab)
#     assert jnp.allclose(rgb, rgb2, atol=1e-3), f"{rgb} != {rgb2}"
# But this test fails:
# for i in range(100):
#     subkey, key = jax.random.split(key)
#     lab = jax.random.uniform(subkey, (3,))
#     lab = lab * jnp.array([100.0, 256.0, 256.0]) - jnp.array([0.0, 128.0, 128.0])
#     rgb = lab_to_rgb(lab)
#     lab2 = rgb_to_lab(rgb)
#     assert jnp.allclose(lab, lab2, atol=4), f"{lab} != {lab2}"
@partial(jnp.vectorize, signature="(k)->(k)")
def lab_to_rgb(lab):
    # LAB to XYZ
    # D65 white point
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])
    y = (lab[0] + 16) / 116
    x = lab[1] / 500 + y
    z = y - lab[2] / 200

    xyz = jnp.stack([x, y, z], axis=-1)
    mask = xyz > 0.2068966
    xyz_cubed = jnp.power(xyz, 3)
    xyz = jnp.where(mask, xyz_cubed, (xyz - 16 / 116) / 7.787)
    xyz = xyz * xyz_ref

    # XYZ to linear RGB
    xyz_to_rgb = jnp.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    rgb = jnp.dot(xyz, xyz_to_rgb.T)

    # Linear RGB to sRGB
    mask = rgb > 0.0031308
    rgb = jnp.where(mask, 1.055 * jnp.power(rgb, 1 / 2.4) - 0.055, 12.92 * rgb)
    rgb = jnp.clip(rgb, 0, 1)

    return rgb

###################
# RGB likelihoods #
###################

laplace = genjax.TFPDistribution(tfp.distributions.Laplace)
class LaplaceRGBPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rgb_scale):
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = laplace.sample(key, lab, rgb_scale)
        rgb = lab_to_rgb(lab2)
        return rgb

    def logpdf(self, observed_rgb, rendered_rgb, rgb_scale):
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = b3d.rgb_to_lab(observed_rgb)
        rgb_logpdf = laplace.logpdf(lab2, lab, rgb_scale)
        return rgb_logpdf

laplace_rgb_pixel_model = LaplaceRGBPixelModel()

class UniformRGBPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb):
        lab = b3d.rgb_to_lab(rendered_rgb)
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        lab2 = genjax.uniform.sample(key, low, high)
        rgb = lab_to_rgb(lab2)
        return rgb

    def logpdf(self, observed_rgb, rendered_rgb):
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = b3d.rgb_to_lab(observed_rgb)
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        rgb_logpdf = genjax.uniform.logpdf(lab2, low, high)
        return rgb_logpdf
uniform_rgb_pixel_model = UniformRGBPixelModel()

class MixtureRGBPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - probs: [p_uniform, *p_centered_at_colors] (N,)
    - rgbs: (N-1, 3) 
    - laplace_scale: () [shared across all laplace dists]
    """
    def sample(self, key, probs, rgbs, laplace_scale):
        key, subkey = jax.random.split(key)
        choice = genjax.categorical.sample(subkey, jnp.log(probs))
        key, subkey = jax.random.split(key)
        uniform_value = uniform_rgb_pixel_model.sample(key, rgbs[0])
        laplace_value = laplace_rgb_pixel_model.sample(subkey, rgbs[choice-1], laplace_scale)
        return jnp.where(choice == 0, uniform_value, laplace_value)
    
    def logpdf(self, observed_rgb, probs, rgbs, laplace_scale):
        uniform_logpdf = uniform_rgb_pixel_model.logpdf(observed_rgb, rgbs[0])
        laplace_logpdfs = jax.vmap(
            lambda rgb: laplace_rgb_pixel_model.logpdf(observed_rgb, rgb, laplace_scale)
        )(rgbs)

        uniform_logpdf = jnp.log(probs[0] + 1e-5) + uniform_logpdf
        laplace_logpdfs = jnp.log(probs[1:] + 1e-5) + laplace_logpdfs
        return jax.scipy.special.logsumexp(jnp.concatenate([uniform_logpdf[None], laplace_logpdfs]))
mixture_rgb_pixel_model = MixtureRGBPixelModel()
mixture_rgb_sensor_model = genjax.map_combinator(in_axes=(0, 0, None))(mixture_rgb_pixel_model)

####################
# RGBD Likelihoods #
####################

class LaplaceRGBDPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - rendered_rgbd (center of laplace dists)
    - rgb_scale (scale for rgb laplace dist, in LAB color space)
    - depth_scale (scale for depth laplace dist)
    """
    def sample(self, key, rendered_rgbd, rgb_scale, depth_scale):
        lab = b3d.rgb_to_lab(rendered_rgbd[..., :3])
        lab2 = laplace.sample(key, lab, rgb_scale)
        rgb = lab_to_rgb(lab2)
        depth = laplace.sample(key, rendered_rgbd[..., 3], depth_scale)
        return jnp.concatenate([rgb, jnp.array([depth])])

    def logpdf(self, observed_rgbd, rendered_rgbd, rgb_scale, depth_scale):
        lab = b3d.rgb_to_lab(rendered_rgbd[..., :3])
        lab2 = b3d.rgb_to_lab(observed_rgbd[..., :3])
        rgb_logpdf = laplace.logpdf(lab2, lab, rgb_scale)
        depth_logpdf = laplace.logpdf(observed_rgbd[..., 3], rendered_rgbd[..., 3], depth_scale)
        return rgb_logpdf + depth_logpdf

laplace_rgbd_pixel_model = LaplaceRGBDPixelModel()

class UniformRGBDPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - rendered_rgbd (used for shape only)
    - mindepth (min value for uniform on depth)
    - maxdepth (max value for uniform on depth)
    """
    def sample(self, key, rendered_rgbd, mindepth, maxdepth):
        lab = b3d.rgb_to_lab(rendered_rgbd[:3])
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        lab2 = genjax.uniform.sample(key, low, high)
        rgb = lab_to_rgb(lab2)
        depth = genjax.uniform.sample(key, jnp.ones_like(rendered_rgbd[3]) * mindepth, jnp.ones_like(rendered_rgbd[3]) * maxdepth)
        return jnp.concatenate([rgb, jnp.array([depth])])

    def logpdf(self, observed_rgbd, rendered_rgbd, mindepth, maxdepth):
        lab = b3d.rgb_to_lab(rendered_rgbd[:3])
        lab2 = b3d.rgb_to_lab(observed_rgbd[:3])
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        rgb_logpdf = genjax.uniform.logpdf(lab2, low, high)
        depth_logpdf = genjax.uniform.logpdf(observed_rgbd[3], jnp.ones_like(rendered_rgbd[3]) * mindepth, jnp.ones_like(rendered_rgbd[3]) * maxdepth)
        return rgb_logpdf + depth_logpdf
    
uniform_rgbd_pixel_model = UniformRGBDPixelModel()

class MixtureRGBDPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - probs: [p_uniform, *p_centered_at_colors] (N,)
    - rgbds: (N-1, 4) 
    - rgb_laplace_scale: () [shared across all laplace dists]
    - depth_laplace_scale: () [shared across all laplace dists]
    - mindepth_for_uniform (min value for uniform on depth)
    - maxdepth_for_uniform (max value for uniform on depth)
    """
    def sample(self, key, probs, rgbds, rgb_laplace_scale, depth_laplace_scale, mindepth_for_uniform, maxdepth_for_uniform):
        key, subkey = jax.random.split(key)
        choice = genjax.categorical.sample(subkey, jnp.log(probs))
        key, subkey = jax.random.split(key)
        uniform_value = uniform_rgbd_pixel_model.sample(key, rgbds[0], mindepth_for_uniform, maxdepth_for_uniform)
        laplace_value = laplace_rgbd_pixel_model.sample(subkey, rgbds[choice-1], rgb_laplace_scale, depth_laplace_scale)
        return jnp.where(choice == 0, uniform_value, laplace_value)
    
    def logpdf(self, observed_rgbd, probs, rgbds, rgb_laplace_scale, depth_laplace_scale, mindepth_for_uniform, maxdepth_for_uniform):
        uniform_logpdf = uniform_rgbd_pixel_model.logpdf(observed_rgbd, rgbds[0], mindepth_for_uniform, maxdepth_for_uniform)
        laplace_logpdfs = jax.vmap(
            lambda rgbd: laplace_rgbd_pixel_model.logpdf(observed_rgbd, rgbd, rgb_laplace_scale, depth_laplace_scale)
        )(rgbds)

        uniform_logpdf = jnp.log(probs[0] + 1e-5) + uniform_logpdf
        laplace_logpdfs = jnp.log(probs[1:] + 1e-5) + laplace_logpdfs
        return jax.scipy.special.logsumexp(jnp.concatenate([uniform_logpdf[None], laplace_logpdfs]))
mixture_rgbd_pixel_model = MixtureRGBDPixelModel()
mixture_rgbd_sensor_model = genjax.map_combinator(in_axes=(0, 0, None, None, None, None))(mixture_rgbd_pixel_model)

