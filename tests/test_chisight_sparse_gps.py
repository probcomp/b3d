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
from genjax import ChoiceMapBuilder as C

import importlib
importlib.reload(ps)


def test_sparse_gps_simulate():

    b3d.rr_init()

    renderer = RendererOriginal()
    key = jax.random.PRNGKey(1000)

    num_timesteps = Pytree.const(10)
    num_particles = Pytree.const(100)
    num_clusters = Pytree.const(4)
    relative_particle_poses_prior_params = (Pose.identity(), .05, 0.25)
    initial_object_poses_prior_params = (Pose.identity(), 2., 0.5)
    camera_pose_prior_params = (Pose.identity(), 0.5, 0.1)
    instrinsics = Pytree.const(b3d.camera.Intrinsics(120, 100, 50., 50., 50., 50., 0.001, 16.))
    sigma_obs = 0.2



    args = (
        (
            num_timesteps, # const object
            num_particles, # const object
            num_clusters, # const object
            relative_particle_poses_prior_params,
            initial_object_poses_prior_params,
            camera_pose_prior_params
        ),
        (instrinsics, sigma_obs)
    )
    trace = ps.sparse_gps_model.simulate(key, args)


    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]

    chm = C["particle_dynamics","state0","object_poses",1].set(Pose.from_translation(jnp.array([0.0, 0.0, 0.1])))
    chm = chm.merge(C["particle_dynamics","state0","initial_camera_pose"].set(Pose.identity()))

    trace,w = ps.sparse_gps_model.importance(
        key,
        chm,
        args,
    )
    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]



    ps.visualize_particle_system(latent_particle_model_args, particle_dynamics_summary, final_state)

    trace.get_sample()(("particle_dynamics","state0","initial_camera_pose"))
    trace.get_sample()(("particle_dynamics","state0","object_poses", 1))

    observation = trace.get_retval()[2]
    sparse_model_args = trace.get_args()[1]

    ps.visualize_sparse_observation(sparse_model_args, observation)
