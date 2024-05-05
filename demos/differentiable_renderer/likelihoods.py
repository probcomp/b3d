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

laplace = genjax.TFPDistribution(tfp.distributions.Laplace)
class LaplaceRGBSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
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

laplace_rgb_sensor_model = LaplaceRGBSensorModel()

class UniformRGBSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
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
uniform_rgb_sensor_model = UniformRGBSensorModel()

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
        uniform_value = uniform_rgb_sensor_model.sample(key, rgbs[0])
        laplace_value = laplace_rgb_sensor_model.sample(subkey, rgbs[choice-1], laplace_scale)
        return jnp.where(choice == 0, uniform_value, laplace_value)
    
    def logpdf(self, observed_rgb, probs, rgbs, laplace_scale):
        uniform_logpdf = uniform_rgb_sensor_model.logpdf(observed_rgb, rgbs[0])
        laplace_logpdfs = jax.vmap(
            lambda rgb: laplace_rgb_sensor_model.logpdf(observed_rgb, rgb, laplace_scale)
        )(rgbs)

        uniform_logpdf = jnp.log(probs[0]) + uniform_logpdf
        laplace_logpdfs = jnp.log(probs[1:]) + laplace_logpdfs
        return jax.scipy.special.logsumexp(jnp.concatenate([uniform_logpdf[None], laplace_logpdfs]))
mixture_rgb_pixel_model = MixtureRGBPixelModel()
mixture_rgb_sensor_model = genjax.map_combinator(in_axes=(0, 0, None))(mixture_rgb_pixel_model)