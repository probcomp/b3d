import importlib

import b3d
import b3d.chisight.particle_system as ps
import jax
import jax.numpy as jnp
from b3d import Pose
from genjax import ChoiceMapBuilder as C
from genjax import Pytree

<<<<<<< HEAD
import importlib

=======
>>>>>>> main
importlib.reload(ps)


def test_sparse_gps_simulate():
    b3d.rr_init()

    key = jax.random.PRNGKey(1000)

    num_timesteps = Pytree.const(10)
    num_particles = Pytree.const(100)
    num_clusters = Pytree.const(4)
    relative_particle_poses_prior_params = (Pose.identity(), 0.1, 0.25)
    initial_object_poses_prior_params = (Pose.identity(), 1.0, 0.5)
    camera_pose_prior_params = (Pose.identity(), 1.0, 0.1)
    instrinsics = Pytree.const(
        b3d.camera.Intrinsics(120, 100, 50.0, 50.0, 50.0, 50.0, 0.001, 16.0)
    )
    sigma_obs = 0.2

    args = (
        (
            num_timesteps,  # const object
            num_particles,  # const object
            num_clusters,  # const object
            relative_particle_poses_prior_params,
            initial_object_poses_prior_params,
            camera_pose_prior_params,
        ),
        (instrinsics, sigma_obs),
    )
    trace = ps.sparse_gps_model.simulate(key, args)

    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]

    chm = C["particle_dynamics", "state0", "object_poses", 1].set(
        Pose.from_translation(jnp.array([0.0, 0.0, 0.1]))
    )
    chm = chm.merge(
        C["particle_dynamics", "state0", "initial_camera_pose"].set(Pose.identity())
    )

<<<<<<< HEAD
    trace, w = ps.sparse_gps_model.importance(
=======
    trace, _w = ps.sparse_gps_model.importance(
>>>>>>> main
        key,
        chm,
        args,
    )
    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]

    ps.visualize_particle_system(
        latent_particle_model_args, particle_dynamics_summary, final_state
    )

    trace.get_sample()(("particle_dynamics", "state0", "initial_camera_pose"))
    trace.get_sample()(("particle_dynamics", "state0", "object_poses", 1))

    observation = trace.get_retval()[2]
    sparse_model_args = trace.get_args()[1]

    ps.visualize_sparse_observation(sparse_model_args, observation)
