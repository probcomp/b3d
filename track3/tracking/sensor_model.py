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

class GaussianRGBDSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, rgb_std, depth_std):
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = genjax.normal.sample(key, lab, rgb_std)
        rgb = lab_to_rgb(lab2)
        depth = genjax.normal.sample(key, rendered_depth, depth_std)
        return (rgb, depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, rgb_std, depth_std):
        observed_rgb, observed_depth = observed
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = b3d.rgb_to_lab(observed_rgb)
        rgb_logpdf = genjax.normal.logpdf(lab2, lab, rgb_std)
        depth_logpdf = genjax.normal.logpdf(observed_depth, rendered_depth, depth_std)
        return rgb_logpdf + depth_logpdf

gaussian_rgbd_sensor_model = GaussianRGBDSensorModel()

laplace = genjax.TFPDistribution(tfp.distributions.Laplace)
class LaplaceRGBDSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, rgb_scale, depth_scale):
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = laplace.sample(key, lab, rgb_scale)
        rgb = lab_to_rgb(lab2)
        depth = laplace.sample(key, rendered_depth, depth_scale)
        return (rgb, depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, rgb_scale, depth_scale):
        observed_rgb, observed_depth = observed
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = b3d.rgb_to_lab(observed_rgb)
        rgb_logpdf = laplace.logpdf(lab2, lab, rgb_scale)
        depth_logpdf = laplace.logpdf(observed_depth, rendered_depth, depth_scale)
        return rgb_logpdf + depth_logpdf

laplace_rgbd_sensor_model = LaplaceRGBDSensorModel()

class UniformRGBDSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, rendered_rgb, rendered_depth, depth_range=(0., 10.)):
        lab = b3d.rgb_to_lab(rendered_rgb)
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        lab2 = genjax.uniform.sample(key, low, high)
        rgb = lab_to_rgb(lab2)
        depth = genjax.uniform.sample(key, jnp.ones_like(rendered_depth) * depth_range[0], jnp.ones_like(rendered_depth) * depth_range[1])
        return (rgb, depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, depth_range=(0., 10.)):
        observed_rgb, observed_depth = observed
        lab = b3d.rgb_to_lab(rendered_rgb)
        lab2 = b3d.rgb_to_lab(observed_rgb)
        low = jnp.ones_like(lab) * jnp.array([0., -128., -128.])
        high = jnp.ones_like(lab) * jnp.array([100., 127., 127.])
        rgb_logpdf = genjax.uniform.logpdf(lab2, low, high)
        depth_logpdf = genjax.uniform.logpdf(observed_depth, jnp.ones_like(rendered_depth) * depth_range[0], jnp.ones_like(rendered_depth) * depth_range[1])
        return rgb_logpdf + depth_logpdf
uniform_rgbd_sensor_model = UniformRGBDSensorModel()

class RGBDMixtureDistribution(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - probs: [p_uniform, p_gaussian, p_laplace]
    - rgb
    - depth
    - submodel_args: [(rgb_std, depth_std), (rgb_scale, depth_scale), (depth_range,)]
    """
    def sample(self, key, probs, rgb, depth, submodel_args):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, 3)
        idx = genjax.categorical.sample(key, probs)
        return [
            uniform_rgbd_sensor_model.sample(keys[0], rgb, depth, *submodel_args[0]),
            gaussian_rgbd_sensor_model.sample(keys[1], rgb, depth, *submodel_args[1]),
            laplace_rgbd_sensor_model.sample(keys[2], rgb, depth, *submodel_args[2]),
        ][idx]

    def logpdf(self, observed, probs, rgb, depth, submodel_args):
        return jnp.logsumexp(jnp.array([
            uniform_rgbd_sensor_model.logpdf(observed, rgb, depth, *submodel_args[0]),
            gaussian_rgbd_sensor_model.logpdf(observed, rgb, depth, *submodel_args[1]),
            laplace_rgbd_sensor_model.logpdf(observed, rgb, depth, *submodel_args[2]),
        ]) + jnp.log(probs))
mixture_rgbd_sensor_model = RGBDMixtureDistribution()

# class MixtureDistribution(genjax.ExactDensity,genjax.JAXGenerativeFunction):
#     """
#     Args:
#     - weights
#     - tuple_of_args_for_submodels
#     """
#     models: list

#     def sample(self, key, weights, *args):
#         key, subkey = jax.random.split(key)
#         keys = jax.random.split(subkey, len(self.models))
#         samples = [model.sample(key, *args[i]) for (i, (model ,key)) in enumerate(zip(self.models, keys))]
#         idx = genjax.categorical.sample(key, weights)
#         return samples[idx]
    
#     def logpdf(self, observed, weights, *args):
#         logpdfs = jnp.array([model.logpdf(observed, *args[i]) for (i, model) in enumerate(self.models)])
#         return jnp.logsumexp(logpdfs + jnp.log(weights))
# mixture_rgbd_sensor_model = MixtureDistribution([
#     UniformRGBDSensorModel(),
#     GaussianRGBDSensorModel(),
#     LaplaceRGBDSensorModel(),
# ])

# def MixtureRGBDSensorModel(
#     p_uniform, p_gaussian, p_laplace,
# ):
#     assert jnp.allclose(p_uniform + p_gaussian + p_laplace, 1.0, atol=1e-4)
#     return MixtureDistribution([
#         UniformRGBDSensorModel(),
#         GaussianRGBDSensorModel(),
#         LaplaceRGBDSensorModel(),
#     ], [p_uniform, p_gaussian, p_laplace]
# )

# def get_rgb_depth_inliers_from_trace(trace):
#     (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
#     model_args = trace.get_args()[1]
#     return get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args)

# def get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args):
#     observed_lab = b3d.rgb_to_lab(observed_rgb)
#     rendered_lab = b3d.rgb_to_lab(rendered_rgb)
#     error = (
#         jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) + 
#         jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
#     )

#     valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

#     color_inliers = (error < model_args.color_tolerance) * valid_data_mask
#     depth_inliers = (jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance) * valid_data_mask
#     inliers = color_inliers * depth_inliers
#     outliers = jnp.logical_not(inliers) * valid_data_mask
#     undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)
#     return (inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask)

# class RGBDSensorModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
#     """
#     sensor_model(rendered_rgb, rendered_depth, color_tolerance, depth_likelihood)
#     """