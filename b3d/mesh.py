import b3d
import jax.numpy as jnp
from b3d import Pose
import jax
import trimesh
from jax.tree_util import register_pytree_node_class

def merge_meshes(meshes):
    vertices = jnp.concatenate([meshes[i].vertices for i in range(len(meshes))])
    vertices_cumsum = jnp.cumsum(jnp.array([0] + [meshes[i].vertices.shape[0] for i in range(len(meshes))]))
    faces = jnp.concatenate([meshes[i].faces + vertices_cumsum[i] for i in range(len(meshes))])
    colors = jnp.concatenate([meshes[i].colors for i in range(len(meshes))])
    return Mesh(vertices, faces, colors)

merge_meshes_jit = jax.jit(merge_meshes)

def transform_and_merge_meshes(meshes, poses):
    vertices = jnp.concatenate([poses[i].apply(meshes[i].vertices) for i in range(len(meshes))])
    vertices_cumsum = jnp.cumsum(jnp.array([0] + [meshes[i].vertices.shape[0] for i in range(len(meshes))]))
    faces = jnp.concatenate([meshes[i].faces + vertices_cumsum[i] for i in range(len(meshes))])
    colors = jnp.concatenate([meshes[i].colors for i in range(len(meshes))])
    return Mesh(vertices, faces, colors)

transform_and_merge_meshes_jit = jax.jit(transform_and_merge_meshes)


def transform_mesh(mesh, pose):
    return Mesh(
        pose.apply(mesh.vertices),
        mesh.faces,
        mesh.colors
    )

transform_mesh_jit = jax.jit(transform_mesh)

@register_pytree_node_class
class Mesh:
    def __init__(self, vertices, faces, colors):
        self.vertices = vertices
        self.faces = faces
        self.colors = colors

    def tree_flatten(self):
        return ((self.vertices, self.faces, self.colors), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def from_obj_file(path):
        trimesh_mesh = trimesh.load_mesh(path)
        vertices = jnp.array(trimesh_mesh.vertices)
        vertices = vertices - jnp.mean(vertices, axis=0)
        faces = jnp.array(trimesh_mesh.faces)
        vertex_colors = jnp.array(trimesh_mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
        return Mesh(vertices, faces, vertex_colors)

    def transform(self, pose):
        return transform_mesh(self, pose)
    
    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices.shape[:-1]}, faces={self.faces.shape[:-1]}, colors={self.colors.shape[:-1]})"

    merge_meshes = staticmethod(merge_meshes)
    merge_meshes_jit = staticmethod(merge_meshes_jit)
    transform_and_merge_meshes = staticmethod(transform_and_merge_meshes)
    transform_and_merge_meshes_jit = staticmethod(transform_and_merge_meshes_jit)
    transform_mesh = staticmethod(transform_mesh)
    transform_mesh_jit = staticmethod(transform_mesh_jit)
