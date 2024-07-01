import b3d
from b3d.renderer.renderer_original import RendererOriginal
import jax
import jax.numpy as jnp
import os
from b3d import Pose, Mesh

import b3d.chisight.particle_system as ps
import genjax
from genjax import Pytree
import jax
from b3d import Pose
import b3d


def test_sparse_gps_simulate():
    renderer = RendererOriginal()
    key = jax.random.PRNGKey(100)

    num_timesteps = Pytree.const(10)
    num_particles = Pytree.const(100)
    num_clusters = Pytree.const(4)
    relative_particle_poses_prior_params = (Pose.identity(), .5, 0.25)
    initial_object_poses_prior_params = (Pose.identity(), 2., 0.5)
    camera_pose_prior_params = (Pose.identity(), 0.1, 0.1)
    instrinsics = Pytree.const(b3d.camera.Intrinsics(120, 100, 50., 50., 50., 50., 0.001, 16.))
    sigma_obs = 0.2


    b3d.rr_init()

    trace = ps.sparse_gps_model.simulate(key, (
        (
            num_timesteps, # const object
            num_particles, # const object
            num_clusters, # const object
            relative_particle_poses_prior_params,
            initial_object_poses_prior_params,
            camera_pose_prior_params
        ),
        (instrinsics, sigma_obs)
    ))


    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]

    # import importlib
    # importlib.reload(b3d.chisight.shared.particle_system)

    ps.visualize_particle_system(latent_particle_model_args, particle_dynamics_summary, final_state)