import jax
import jax.numpy as jnp
import genjax
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as rendering
import b3d
from b3d import Pose
from b3d.model import uniform_pose
import rerun as rr
import demos.differentiable_renderer.tracking.utils as utils

def normalize(v):
    return v / jnp.sum(v)

def single_object_model_factory(
        renderer, likelihood,
        renderer_hyperparams, get_example_rgbd
    ):
    """
    Args:
    - renderer
    - likelihood
        Should be a distribution on images.
        Should accept (weights, attributes, *likelihood_args) as input.
    - renderer_hyperparams
    - get_example_rgbd
        Should accept (weights, attributes, likelihood_args) as input
        and return an rgbd image.
    """
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors, likelihood_args):
        X_WO = uniform_pose(jnp.ones(3)*-10.0, jnp.ones(3)*10.0) @ "pose"
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
        
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, renderer_hyperparams
        )

        print(weights.shape, attributes.shape)
        print(likelihood_args)
        print(likelihood)
        observed_rgbd = likelihood(weights, attributes) @ "observed_rgbd"
        # example_rgbd = get_example_rgbd(weights, attributes, likelihood_args)
        return (observed_rgbd, weights, attributes)
        return observed_rgbd
    return model
