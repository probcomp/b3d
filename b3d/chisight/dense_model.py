import genjax
import jax
import jax.numpy as jnp
import b3d
import trimesh
import os
import rerun as rr

import importlib
importlib.reload(b3d)
import b3d


### Multiple object model ###
def multiple_object_model_factory(
        renderer, likelihood,
        renderer_hyperparams
    ):
    """
    Args:
    - renderer
    - likelihood
        Should be a distribution on images.
        Should accept (weights, attributes, *likelihood_args) as input.
    - renderer_hyperparams
    """
    @genjax.static_gen_fn
    def model(X_WC, Xs_WO, vertices_O, faces, vertex_colors):
        # W = world frame; C = camera frame; O = object frame

        #    (N, V, 3)    (N, F, 3)   (N, V, 3)
        # where N = num objects
        N = vertices_O.shape[0]

        vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W.reshape(-1, 3))
        
        v = vertices_C.reshape(-1, 3)
        vc = vertex_colors.reshape(-1, 3)

        # shift up each object's face indices by the number of vertices in the previous objects
        f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
        f = f.reshape(-1, 3)

        # weights, attributes = rendering.render_to_rgbd_dist_params(
        #     renderer, v, f, vc, renderer_hyperparams
        # )
        # observed_rgbd = likelihood(weights, attributes) @ "observed_rgbd"

        observed_rgbd, likelihood_metadata = likelihood(renderer, v, f, vc, renderer_hyperparams)

        return (observed_rgbd, likelihood_metadata)
    return model
