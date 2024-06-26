import b3d
from b3d.renderer.renderer_original import RendererOriginal
from b3d.chisight.dense.dense_likelihood import make_dense_image_likelihood_from_renderer, DenseImageLikelihoodArgs
import jax
import jax.numpy as jnp
import os
from b3d import Pose, Mesh

import b3d.chisight.shared.particle_system as ps
import genjax
from genjax import Pytree
import jax
from b3d import Pose
import b3d



renderer = RendererOriginal()
dense_image_likelihood = make_dense_image_likelihood_from_renderer(renderer)

likelihood_args = DenseImageLikelihoodArgs(1.0, 1.0, 1.0, 1.0, 1.0)
key = jax.random.PRNGKey(10)
mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = Mesh.from_obj_file(mesh_path)
meshes = [mesh]
poses = [Pose(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]))]

logpdf = dense_image_likelihood.logpdf(
    jnp.zeros((renderer.height, renderer.width, 4)), poses, meshes, likelihood_args
)


key = jax.random.PRNGKey(125)

num_timesteps = Pytree.const(4)
num_particles = Pytree.const(5)
num_clusters = Pytree.const(3)
relative_particle_poses_prior_params = (Pose.identity(), .5, 0.25)
initial_object_poses_prior_params = (Pose.identity(), 2., 0.5)
camera_pose_prior_params = (Pose.identity(), 0.1, 0.1)
instrinsics = Pytree.const(b3d.camera.Intrinsics(120, 100, 50., 50., 50., 50., 0.001, 16.))
sigma_obs = 0.2
