import b3d
from b3d.renderer.renderer_original import RendererOriginal
from b3d.chisight.dense.dense_likelihood import make_dense_image_likelihood_from_renderer, DenseImageLikelihoodArgs
import jax
import jax.numpy as jnp
import os
from b3d import Pose, Mesh

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
