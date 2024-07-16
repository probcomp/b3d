"""
This file defines several likelihoods over RGBD images.
"""

import jax.numpy as jnp
import jax
import jax.tree_util as jtu
import genjax
import b3d
from genjax import Pytree
from tensorflow_probability.substrates import jax as tfp

import b3d.modeling_utils

def normalize(l):
    return jnp.where(
        jnp.sum(l, axis=-1) < 1e-6,
        jnp.ones_like(l) / l.shape[-1],
        l / (jnp.sum(l, axis=-1)[..., None] + 1e-8)
    )

@Pytree.dataclass
class ArgMap(genjax.ExactDensity):
    """
    `ArgMap(dist : genjax.ExactDensity, argmap : function)(*args)`
    is the distribution which samples from `dist(*argmap(*args))`.
    """
    dist : any  = genjax.Pytree.static()
    argmap : any = genjax.Pytree.static()

    def sample(self, key, *args):
        return self.dist.sample(key, *self.argmap(*args))
    
    def logpdf(self, observed, *args):
        return self.dist.logpdf(observed, *self.argmap(*args))

@Pytree.dataclass
class ImageDistFromPixelDist(genjax.ExactDensity):
    """
    Given a distribution on a pixel's value, returns a distribution on an image.
    Constructor args:
    - pixel_dist: Dist on pixel value (e.g. on an RGB or RGBD value)
    - map_args: [bool] (N,) whether the nth arg should be mapped over
        along the width and height axes (otherwise all pixels will have
        the same value for that arg)
    Distribution args:
    - height
    - width 
    - *pixel_dist_args
    """
    pixel_dist : any = genjax.Pytree.static()
    map_args : any = genjax.Pytree.static()

    def _flattened_args(self, args):
        return [
            jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), arg) if do_map else arg
            for (arg, do_map) in zip(args, self.map_args)
        ]
    def _vmap_in_axes(self):
        return (0, *[0 if a else None for a in self.map_args])
    
    def sample(self, key, height, width, *pixel_dist_args):
        keys = jax.random.split(key, height*width)
        
        pixels = jax.vmap(
            lambda key, *args: self.pixel_dist.sample(key, *args),
            in_axes=self._vmap_in_axes()
        )(keys, *(self._flattened_args(pixel_dist_args)))
        return pixels.reshape((height, width, -1))

    def logpdf(self, observed_image, height, width, *pixel_dist_args):
        if observed_image.shape[:2] != (height, width):
            print("Warning: unequal shapes in ImageDistFromPixelDist.logpdf")
            return -jnp.inf
        logpdfs = jax.vmap(
            lambda pixel, *args: self.pixel_dist.logpdf(pixel, *args),
            in_axes=self._vmap_in_axes()
        )(observed_image.reshape(-1, observed_image.shape[-1]), *self._flattened_args(pixel_dist_args))
        return logpdfs.sum()

@Pytree.dataclass
class UniformRGBDPixelModel(genjax.ExactDensity):
    """
    Args:
    - mindepth () (min value for uniform on depth)
    - maxdepth () (max value for uniform on depth)
    Returns:
    - rgbd (4,)
    """
    def sample(self, key, mindepth, maxdepth):
        low = jnp.zeros(3)
        high = jnp.ones(3)
        rgb = b3d.modeling_utils.uniform.sample(key, low, high)
        depth = b3d.modeling_utils.uniform.sample(key, mindepth, maxdepth)
        return jnp.concatenate([rgb, jnp.array([depth])])

    def logpdf(self, observed_rgbd, mindepth, maxdepth):
        return (
            b3d.modeling_utils.uniform.logpdf(observed_rgbd[:3], jnp.zeros(3), jnp.ones(3)) +
            b3d.modeling_utils.uniform.logpdf(observed_rgbd[3], mindepth, maxdepth)
        )
        
uniform_rgbd_pixel_model = UniformRGBDPixelModel()

uniform_rgbd_image_model = ImageDistFromPixelDist(uniform_rgbd_pixel_model, [False, False])
image_sample = uniform_rgbd_image_model.sample(jax.random.PRNGKey(0), 120, 100, 0.1, 0.9)
score = uniform_rgbd_image_model.logpdf(image_sample, 120, 100, 0.1, 0.9)
# assert jnp.abs(score - 120*100 * (-jnp.log(.9 - .1))) < 1e-3
# assert image_sample.shape == (120, 100, 4)
# assert image_sample[:, :, 3].min() >= 0.1
# assert image_sample[:, :, 3].max() <= 0.9
# assert image_sample[:, :, :3].min() >= 0
# assert image_sample[:, :, :3].max() <= 1


laplace = b3d.modeling_utils.tfp_distribution(tfp.distributions.Laplace)

@Pytree.dataclass
class RGBDPixelModel(genjax.ExactDensity):
    """
    Combine a model on an RGB value and a model on a depth value, each
    centered at a given value, into a model on a RGB-Depth value.
    
    Constructor args:
    - depth_pixel_model : genjax.ExactDensity
    - color_pixel_model : genjax.ExactDensity

    Distribution args:
    - rendered_rgbd : a (4,) array around which the RGBD value will be sampled
    - depth_args : tuple of arguments so that `(rgb, *depth_args)` is accepted
        by `depth_pixel_model`
    - color_args : tuple of arguments so that `(rgb, *color_args)` is accepted
        by `color_pixel_model`
    """
    depth_pixel_model : any = genjax.Pytree.static()
    color_pixel_model : any = genjax.Pytree.static()

    def sample(self, key, rendered_rgbd, depth_args, color_args):
        key, subkey = jax.random.split(key)
        color = self.color_pixel_model.sample(key, rendered_rgbd[:3], *color_args)
        depth = self.depth_pixel_model.sample(subkey, rendered_rgbd[3], *depth_args)
        return jnp.concatenate([color, jnp.array([depth])])
    
    def logpdf(self, observed_rgbd, rendered_rgbd, depth_args, color_args):
        rgb_logpdf = self.color_pixel_model.logpdf(observed_rgbd[:3], rendered_rgbd[:3], *color_args)
        depth_logpdf = self.depth_pixel_model.logpdf(observed_rgbd[3], rendered_rgbd[3], *depth_args)
        return rgb_logpdf + depth_logpdf

laplace_rgbd_pixel_model = RGBDPixelModel(laplace, laplace)
laplace_rgb_uniform_depth_pixel_model = RGBDPixelModel(
    ArgMap(b3d.modeling_utils.uniform, lambda r, low, high: (low, high)),
    laplace
)
uniform_rgb_laplace_depth_pixel_model = RGBDPixelModel(
    laplace,
    ArgMap(b3d.modeling_utils.uniform, lambda r: (jnp.zeros(3), jnp.ones(3)))
)
s1 = laplace_rgbd_pixel_model.sample(jax.random.PRNGKey(0), jnp.array([0.5, 0.5, 0.5, 1.5]), (0.1,), (0.9,))
score = laplace_rgbd_pixel_model.logpdf(s1, jnp.array([0.5, 0.5, 0.5, 1.5]), (0.1,), (0.9,))
assert score > -jnp.inf
s2 = laplace_rgb_uniform_depth_pixel_model.sample(jax.random.PRNGKey(0), jnp.array([0.5, 0.5, 0.5, 1.5]), (0., 2.0), (0.9,))
score = laplace_rgb_uniform_depth_pixel_model.logpdf(s2, jnp.array([0.5, 0.5, 0.5, 1.5]), (0., 2.0), (0.9,))
assert score > -jnp.inf
s3 = uniform_rgb_laplace_depth_pixel_model.sample(jax.random.PRNGKey(0), jnp.array([0.5, 0.5, 0.5, 1.5]), (0.1,), ())
score = uniform_rgb_laplace_depth_pixel_model.logpdf(s3, jnp.array([0.5, 0.5, 0.5, 1.5]), (0.1,), ())
assert score > -jnp.inf

@Pytree.dataclass
class VmapMixturePixelModel(genjax.ExactDensity):
    """
    A distribution which is a mixture of `dist` on different arguments.
    
    Constructor args:
    - dist : genjax.ExactDensity

    Distribution args:
    - probs : (N,) array of probabilities
    - *args : list of arguments so that `dist.sample(key, *args[i])` is valid
        for each i
    """
    dist : any = genjax.Pytree.static()

    def sample(self, key, probs, *args):
        key, subkey = jax.random.split(key)
        component = genjax.categorical.sample(subkey, jnp.log(probs))
        return self.dist.sample(key, *[jtu.tree_map(lambda x: x[component], a) for a in args])
    
    def logpdf(self, observed, probs, *args):
        logprobs = jax.vmap(
            lambda component: self.dist.logpdf(observed, *[jtu.tree_map(lambda x: x[component], a) for a in args])
        )(jnp.arange(probs.shape[0]))
        # jax.debug.print("logprobs = {logprobs}; probs = {probs}", logprobs=logprobs, probs=probs)
        return jax.scipy.special.logsumexp(logprobs + jnp.log(probs + 1e-3))

multilaplace_pixel_model = ArgMap(
    VmapMixturePixelModel(laplace_rgbd_pixel_model),
    lambda probs, rgbds, depth_scale, color_scale: (
        probs, rgbds,
        (depth_scale[0] * jnp.ones(probs.size),),
        (color_scale[0] * jnp.ones(probs.size),)
    ))
s4 = multilaplace_pixel_model.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.8]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    (0.1,), (0.9,)
)
score = multilaplace_pixel_model.logpdf(
    s4,
    jnp.array([0.2, 0.8]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    (0.1,), (0.9,)
)
assert score > -jnp.inf

multi_uniform_rgb_depth_laplace = ArgMap(
    VmapMixturePixelModel(uniform_rgb_laplace_depth_pixel_model),
    lambda probs, rgbds, depth_scale: (
        probs, rgbds,
        (depth_scale * jnp.ones(probs.shape[0]),),
        ()
    ))
multi_uniform_rgb_depth_laplace.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.8]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    0.1
)

@Pytree.dataclass
class PythonMixturePixelModel(genjax.ExactDensity):
    """
    Mixture of different distributions.
    Constructor:
    - dists : list of genjax.ExactDensity

    Distribution args:
    - probs : (N,) array of probabilities
    - args : list of arguments so that `dists[i].sample(key, *args[i])` is valid
        for each i
    """
    dists : any = genjax.Pytree.static()

    def sample(self, key, probs, args):
        values = []
        for (i, dist) in enumerate(self.dists):
            key, subkey = jax.random.split(key)
            values.append(dist.sample(subkey, *args[i]))
        values = jnp.array(values)
        key, subkey = jax.random.split(key)
        component = genjax.categorical.sample(subkey, jnp.log(probs))
        return values[component]
    
    def logpdf(self, observed, probs, args):
        logprobs = []
        for (i, dist) in enumerate(self.dists):
            lp = dist.logpdf(observed, *args[i])
            assert lp.shape == ()
            logprobs.append(lp)
        logprobs = jnp.stack(logprobs)
        # jax.debug.print("lps = {lps}", lps=logprobs)
        return jax.scipy.special.logsumexp(logprobs) #  + jnp.log(probs + 1e-6), axis=0)

uniform_multilaplace_mixture = ArgMap(
    PythonMixturePixelModel([uniform_rgbd_pixel_model, multilaplace_pixel_model]),
    lambda probs, rgbds, depth_scale, color_scale, mindepth, maxdepth: (
        jnp.array([probs[0], jnp.sum(probs[1:])]),
        [ (mindepth, maxdepth),
          (normalize(probs[1:]), rgbds, depth_scale, color_scale) ]
    )
)

mixture_of_uniform_and_multi_uniformrgb_laplacedepth = ArgMap(
    PythonMixturePixelModel([uniform_rgbd_pixel_model, multi_uniform_rgb_depth_laplace]),
    lambda probs, rgbds, depth_scale, mindepth, maxdepth: (
        jnp.array([probs[0], jnp.sum(probs[1:])]),
        [ (mindepth, maxdepth),
          (normalize(probs[1:]), rgbds, depth_scale) ]
    ))

s5 = uniform_multilaplace_mixture.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    (0.1,), (0.1,), 0.0, 2.0
)
score = uniform_multilaplace_mixture.logpdf(
    s5,
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    (0.1,), (0.1,), 0.0, 2.0
)
assert score > -jnp.inf

s6 = mixture_of_uniform_and_multi_uniformrgb_laplacedepth.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    0.1, 0.0, 2.0
)
score = mixture_of_uniform_and_multi_uniformrgb_laplacedepth.logpdf(
    s6,
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    0.1, 0.0, 2.0
)
assert score > -jnp.inf

def get_uniform_multilaplace_image_dist_with_fixed_params(
        height, width, depth_scale, color_scale, mindepth, maxdepth
    ):
    """
    Get a distribution on images in which each pixel's value is generated
    independently, from a distribution on RGBD values which is a mixture
    of a uniform and N laplace distributions with different centers.

    Function arguments:
    - height
    - width
    - depth_scale : () scale for the laplaces on depth
    - color_scale : () scale for the laplaces on RGB
    - mindepth : () min value for uniform on depth
    - maxdepth : () max value for uniform on depth

    Distribution arguments:
    - weights : (N + 1,) array of probabilities. The first weight is the probability
        assigned to the uniform distribution.
    - rgbds : (N, 4) array of RGBD values: the centers for the N laplace distributions.
    """
    return ArgMap(
        ImageDistFromPixelDist(
            uniform_multilaplace_mixture,
            [True, True, False, False, False, False]
        ),
        lambda weights, rgbds: ( height, width, weights, rgbds,
                                (depth_scale,), (color_scale,), mindepth, maxdepth )
    )

### Some RGB only likelihoods:
multilaplace_pixel_model_rgb_only = ArgMap(
    VmapMixturePixelModel(laplace),
    lambda probs, rgbs, color_scale: (
        probs, rgbs,
        (color_scale * jnp.ones(probs.size),)
    ))

uniform_multilaplace_mixture_rgb_only = ArgMap(
    PythonMixturePixelModel([b3d.modeling_utils.uniform, multilaplace_pixel_model_rgb_only]),
    lambda probs, rgbs, color_scale: (
        jnp.array([probs[0], jnp.sum(probs[1:])]),
        [ (jnp.zeros(3), jnp.ones(3)),
          (normalize(probs[1:]), rgbs, color_scale) ]
    )
)

s1 = multilaplace_pixel_model_rgb_only.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.8]),
    jnp.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]),
    0.1
)
score = multilaplace_pixel_model_rgb_only.logpdf(
    s1,
    jnp.array([0.2, 0.8]),
    jnp.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]),
    0.1
)
assert score > -jnp.inf

s2 = uniform_multilaplace_mixture_rgb_only.sample(
    jax.random.PRNGKey(0),
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]),
    0.1
)
score = uniform_multilaplace_mixture_rgb_only.logpdf(
    s2,
    jnp.array([0.2, 0.3, 0.5]),
    jnp.array([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]),
    0.1
)
assert score > -jnp.inf

def get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
        height, width, color_scale
    ):
    """
    Get a distribution on images in which each pixel's value is generated
    independently, from a distribution on RGB values which is a mixture
    of a uniform and N laplace distributions with different centers.

    Function arguments:
    - height
    - width
    - color_scale : () scale for the laplaces on RGB
    
    Distribution arguments:
    - weights : (N + 1,) array of probabilities. The first weight is the probability
        assigned to the uniform distribution.
    - rgbds : (N, 4) array of RGBD values: the centers for the N laplace distributions.
    """
    return ArgMap(
        ImageDistFromPixelDist(
            uniform_multilaplace_mixture_rgb_only,
            [True, True, False]
        ),
        lambda weights, rgbs: ( height, width, weights, rgbs, color_scale)
    )


### K Rays Likelihood ###
