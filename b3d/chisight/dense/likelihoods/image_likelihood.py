import genjax
from genjax.generative_functions.distributions import ExactDensity
import jax.numpy as jnp
import b3d
from b3d import Mesh, Pose
from collections import namedtuple
from b3d.modeling_utils import uniform_discrete, uniform_pose
import jax
import os

from genjax import Pytree

def make_image_likelihood(intermediate_func, renderer):
    @Pytree.dataclass
    class ImageLikelihood(genjax.ExactDensity):
        def sample(self, key, scene_mesh, likelihood_args):
            rendered_rgbd = renderer.render_rgbd(
                scene_mesh.vertices,
                scene_mesh.faces,
                scene_mesh.vertex_attributes,
            )
            return rendered_rgbd

        def logpdf(self, observed_rgbd, scene_mesh, likelihood_args):
            results = intermediate_func(observed_rgbd, scene_mesh, renderer, likelihood_args)
            return results["score"]
        
    image_likelihood = ImageLikelihood()
    return image_likelihood