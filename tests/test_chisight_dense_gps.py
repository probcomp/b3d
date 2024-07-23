import b3d
from b3d.renderer.renderer_original import RendererOriginal
from b3d.chisight.dense.likelihoods import (
    KRaysImageLikelihoodArgs,
    make_krays_image_observation_model,
)
import jax
import jax.numpy as jnp
from b3d import Pose, Mesh

import b3d.chisight.particle_system as ps
from genjax import Pytree

import importlib

importlib.reload(ps)


def test_dense_gps_model():
    b3d.rr_init()

    renderer = RendererOriginal()
    key = jax.random.PRNGKey(10)

    num_timesteps = Pytree.const(10)
    num_particles = Pytree.const(100)
    num_clusters = Pytree.const(4)
    relative_particle_poses_prior_params = (Pose.identity(), 0.35, 0.25)
    initial_object_poses_prior_params = (Pose.identity(), 2.0, 0.5)
    camera_pose_prior_params = (Pose.identity(), 0.5, 0.1)
    instrinsics = Pytree.const(
        b3d.camera.Intrinsics(120, 100, 50.0, 50.0, 50.0, 50.0, 0.001, 16.0)
    )
    sigma_obs = 0.2

    trace = ps.sparse_gps_model.simulate(
        key,
        (
            (
                num_timesteps,  # const object
                num_particles,  # const object
                num_clusters,  # const object
                relative_particle_poses_prior_params,
                initial_object_poses_prior_params,
                camera_pose_prior_params,
            ),
            (instrinsics, sigma_obs),
        ),
    )

    def cube_mesh_with_size_and_color(size, color):
        mesh = Mesh.cube_mesh(size)
        mesh.vertex_attributes = jnp.ones_like(mesh.vertices) * color
        return mesh

    meshes = jax.vmap(cube_mesh_with_size_and_color)(
        jnp.ones((num_particles.const, 3)) * jnp.array([[0.1, 0.1, 0.01]]),
        jax.random.uniform(key, (num_particles.const, 3)),
    )

    likelihood = make_krays_image_observation_model(renderer)
    dense_gps_model = ps.make_dense_gps_model(likelihood)
    dense_likelihood_args = KRaysImageLikelihoodArgs(1.0, 1.0, 1.0, 1.0, 1.0)
    dense_gps_args = (meshes, dense_likelihood_args)
    trace = dense_gps_model.simulate(
        key,
        (
            (
                num_timesteps,  # const object
                num_particles,  # const object
                num_clusters,  # const object
                relative_particle_poses_prior_params,
                initial_object_poses_prior_params,
                camera_pose_prior_params,
            ),
            dense_gps_args,
        ),
    )

    particle_dynamics_summary = trace.get_retval()[0]
    final_state = trace.get_retval()[1]
    latent_particle_model_args = trace.get_args()[0]
    meshes = trace.get_args()[1][0]

    (
        num_timesteps,  # const object
        num_particles,  # const object
        num_clusters,  # const object
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params,
    ) = latent_particle_model_args

    ps.visualize_particle_system(
        latent_particle_model_args, particle_dynamics_summary, final_state
    )
    ps.visualize_dense_gps(
        latent_particle_model_args,
        dense_gps_args,
        particle_dynamics_summary,
        final_state,
    )

    ps.visualize_dense_observation(trace.get_choices()["obs", "image"])
