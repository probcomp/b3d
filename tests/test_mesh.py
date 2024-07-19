import b3d
from b3d import Pose
from b3d import Mesh
import jax.numpy as jnp
import jax
import os
import unittest
from b3d.renderer.renderer_original import RendererOriginal

class MeshTests(unittest.TestCase):

    def test_merge_meshes(self):
        mesh1 = Mesh(
            jax.random.normal(jax.random.PRNGKey(0), (50, 3)),
            jnp.zeros((50,3), dtype=jnp.int32),
            jax.random.uniform(jax.random.PRNGKey(1), (50, 3))
        )
        mesh2 = Mesh(
            jax.random.normal(jax.random.PRNGKey(0), (50, 3)),
            jnp.zeros((50,3), dtype=jnp.int32),
            jax.random.uniform(jax.random.PRNGKey(1), (50, 3))
        )

        Mesh.merge_meshes([mesh1, mesh2])
        combined_mesh = Mesh.merge_meshes([mesh1, mesh2])
        assert combined_mesh.vertices.shape[0] == mesh1.vertices.shape[0] + mesh2.vertices.shape[0]
        assert combined_mesh.vertex_attributes.shape[0] == mesh1.vertex_attributes.shape[0] + mesh2.vertex_attributes.shape[0]
        assert combined_mesh.faces.shape[0] == mesh1.faces.shape[0] + mesh2.faces.shape[0]
        assert (combined_mesh.vertices == jnp.concatenate([mesh1.vertices, mesh2.vertices])).all()
        assert (combined_mesh.vertex_attributes == jnp.concatenate([mesh1.vertex_attributes, mesh2.vertex_attributes])).all()

    def test_transform_mesh(self):
        mesh = Mesh(
            jax.random.normal(jax.random.PRNGKey(0), (50, 3)),
            jnp.zeros((50,3), dtype=jnp.int32),
            jax.random.uniform(jax.random.PRNGKey(1), (50, 3))
        )
        pose = Pose.from_translation(
            jax.random.normal(jax.random.PRNGKey(0), (3,)),
        )
        transformed_mesh = Mesh.transform_mesh(mesh, pose)
        assert (transformed_mesh.vertices == pose.apply(mesh.vertices)).all()
        assert (transformed_mesh.vertex_attributes == mesh.vertex_attributes).all()
        assert (transformed_mesh.faces == mesh.faces).all()

    def test_vmapping_mesh(self):
        import importlib
        importlib.reload(b3d.mesh)
        importlib.reload(b3d.renderer.renderer_original)
        importlib.reload(b3d.utils)

        many_cubes_mesh = jax.vmap(Mesh.cube_mesh)(jnp.ones((100,3)))
        merged_mesh = Mesh.merge_meshes(many_cubes_mesh)

        b3d.rr_init()
        renderer = b3d.renderer.renderer_original.RendererOriginal()

        m = merged_mesh.transform(Pose.from_translation(jnp.array([0, 0, 2.1])))

        rgbd = renderer.render_rgbd_from_mesh(m)
        b3d.utils.rr_log_rgbd("image", rgbd)
