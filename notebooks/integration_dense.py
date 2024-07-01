import b3d
from b3d.renderer.renderer_original import RendererOriginal
from b3d.chisight.dense.likelihoods import KRaysImageLikelihoodArgs, make_krays_image_observation_model, get_rgb_depth_inliers_from_observed_rendered_args
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


import importlib
importlib.reload(ps)

renderer = RendererOriginal()
key = jax.random.PRNGKey(10)

num_timesteps = Pytree.const(4)
num_particles = Pytree.const(5)
num_clusters = Pytree.const(3)
relative_particle_poses_prior_params = (Pose.identity(), .5, 0.25)
initial_object_poses_prior_params = (Pose.identity(), 2., 0.5)
camera_pose_prior_params = (Pose.identity(), 0.1, 0.1)
instrinsics = Pytree.const(b3d.camera.Intrinsics(120, 100, 50., 50., 50., 50., 0.001, 16.))
sigma_obs = 0.2

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

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = Mesh.from_obj_file(mesh_path)
batched_mesh = jax.tree.map(lambda arr: jnp.tile(arr[None, ...], (num_particles.const, 1)), mesh)

likelihood = make_krays_image_observation_model(renderer)
dense_gps_model = ps.make_dense_gps_model(likelihood)
dense_likelihood_args = KRaysImageLikelihoodArgs(1.0, 1.0, 1.0, 1.0, 1.0)

trace = dense_gps_model.simulate(key, (
    (
        num_timesteps, # const object
        num_particles, # const object
        num_clusters, # const object
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params
    ),
    (batched_mesh, dense_likelihood_args)
))
