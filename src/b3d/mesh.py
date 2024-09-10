import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import trimesh
from jax.tree_util import register_pytree_node_class

import b3d


@jax.jit
def merge_meshes(meshes):
    vertices = jnp.concatenate([meshes[i].vertices for i in range(len(meshes))])
    vertices_cumsum = jnp.cumsum(
        jnp.array([0] + [meshes[i].vertices.shape[0] for i in range(len(meshes))])
    )
    faces = jnp.concatenate(
        [meshes[i].faces + vertices_cumsum[i] for i in range(len(meshes))]
    )
    vertex_attributes = jnp.concatenate(
        [meshes[i].vertex_attributes for i in range(len(meshes))]
    )
    return Mesh(vertices, faces, vertex_attributes)


@jax.jit
def transform_and_merge_meshes(meshes, poses):
    vertices = jnp.concatenate(
        [poses[i].apply(meshes[i].vertices) for i in range(len(meshes))]
    )
    vertices_cumsum = jnp.cumsum(
        jnp.array([0] + [meshes[i].vertices.shape[0] for i in range(len(meshes))])
    )
    faces = jnp.concatenate(
        [meshes[i].faces + vertices_cumsum[i] for i in range(len(meshes))]
    )
    vertex_attributes = jnp.concatenate(
        [meshes[i].vertex_attributes for i in range(len(meshes))]
    )
    return Mesh(vertices, faces, vertex_attributes)


@jax.jit
def squeeze_mesh(mesh):
    vertices = jnp.concatenate(mesh.vertices)
    vertices_cumsum = jnp.arange(mesh.vertices.shape[0]) * mesh.vertices.shape[1]
    faces = jnp.concatenate(mesh.faces + vertices_cumsum[:, None, None])
    vertex_attributes = jnp.concatenate(mesh.vertex_attributes)
    full_mesh = b3d.mesh.Mesh(vertices, faces, vertex_attributes)
    return full_mesh


@jax.jit
def transform_mesh(mesh, pose):
    return Mesh(pose.apply(mesh.vertices), mesh.faces, mesh.vertex_attributes)


def rr_visualize_mesh(channel, mesh):
    rr.log(
        channel,
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_colors=mesh.vertex_attributes,
        ),
    )


@jax.jit
def mesh_from_xyz_colors_dimensions(xyz, colors, dimensions):
    meshes = b3d.mesh.transform_mesh(
        jax.vmap(b3d.mesh.Mesh.cube_mesh)(dimensions, colors),
        b3d.Pose.from_translation(xyz)[:, None],
    )
    return b3d.mesh.Mesh.squeeze_mesh(meshes)


@jax.jit
def plane_mesh_from_plane_and_dimensions(pose, w, h, color):
    vertices = jnp.array(
        [
            [-w / 2, -h / 2, 0],
            [-w / 2, h / 2, 0],
            [w / 2, h / 2, 0],
            [w / 2, -h / 2, 0],
        ]
    )
    vertices = pose.apply(vertices)
    faces = jnp.array(
        [
            [0, 1, 3],
            [3, 1, 2],
        ]
    )
    vertex_attributes = jnp.ones((len(vertices), 3)) * color
    return Mesh(vertices, faces, vertex_attributes)


@register_pytree_node_class
class Mesh:
    def __init__(self, vertices, faces, vertex_attributes):
        """The Mesh class represents a 3D triangle mesh with the shape represented in terms
        of 3D vertices and faces that form triangles from the vertices. And, the vertex_attributes
        are associated with each of the vertices.

        vertices: jnp.ndarray of shape (N, 3) representing the 3D coordinates of the vertices.
        faces: jnp.ndarray of shape (M, 3) representing the indices of the vertices that form the faces.
        vertex_attributes: jnp.ndarray of shape (N, D) representing the attributes of the vertices.
        """
        self.vertices = vertices
        self.faces = faces
        self.vertex_attributes = vertex_attributes

    def tree_flatten(self):
        return ((self.vertices, self.faces, self.vertex_attributes), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(mesh):
        return Mesh(
            jnp.copy(mesh.vertices),
            jnp.copy(mesh.faces),
            jnp.copy(mesh.vertex_attributes),
        )

    @staticmethod
    def from_obj_file(path):
        trimesh_mesh = trimesh.load_mesh(path, process=False, validate=False)
        return Mesh.from_trimesh(trimesh_mesh)

    from_obj = staticmethod(from_obj_file)

    @staticmethod
    def from_trimesh(trimesh_mesh):
        vertices = jnp.array(trimesh_mesh.vertices)
        faces = jnp.array(trimesh_mesh.faces)
        if not isinstance(trimesh_mesh.visual, trimesh.visual.color.ColorVisuals):
            vertex_colors = (
                jnp.array(trimesh_mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
            )
        else:
            vertex_colors = (
                jnp.array(trimesh_mesh.visual.vertex_colors)[..., :3] / 255.0
            )
        return Mesh(vertices, faces, vertex_colors)

    def save(self, filename):
        trimesh_mesh = trimesh.Trimesh(
            self.vertices,
            self.faces,
            vertex_colors=np.array(self.vertex_attributes * 255).astype(np.uint8),
        )
        with open(filename, "w") as f:
            f.write(
                trimesh.exchange.obj.export_obj(
                    trimesh_mesh, include_normals=True, include_texture=True
                )
            )

    def transform(self, pose):
        return transform_mesh(self, pose)

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.vertices.shape[:-1]}, faces={self.faces.shape[:-1]}, vertex_attributes={self.vertex_attributes.shape[:-1]})"

    def __len__(self):
        # assert len(self.vertices.shape) == 3, "This is not a batched mesh object."
        return self.vertices.shape[0]

    def __getitem__(self, index):
        return Mesh(
            self.vertices[index], self.faces[index], self.vertex_attributes[index]
        )

    merge_meshes = staticmethod(merge_meshes)
    transform_and_merge_meshes = staticmethod(transform_and_merge_meshes)
    transform_mesh = staticmethod(transform_mesh)
    squeeze_mesh = staticmethod(squeeze_mesh)
    mesh_from_xyz_colors_dimensions = staticmethod(mesh_from_xyz_colors_dimensions)

    def rr_visualize(self, channel):
        rr_visualize_mesh(channel, self)

    def scale(self, scale) -> "Mesh":
        self.vertices = self.vertices.at[:, 0].multiply(scale[0])
        self.vertices = self.vertices.at[:, 1].multiply(scale[1])
        self.vertices = self.vertices.at[:, 2].multiply(scale[2])
        return Mesh(self.vertices, self.faces, self.vertex_attributes)

    @staticmethod
    def cube_mesh(dimensions=jnp.ones(3), color=jnp.array([1.0, 0.0, 0.0])):
        vertices = (
            jnp.array(
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                ]
            )
            * dimensions[None, ...]
        )
        faces = jnp.array(
            [
                [1, 19, 12],
                [1, 6, 19],
                [0, 9, 7],
                [0, 4, 9],
                [8, 23, 18],
                [8, 11, 23],
                [14, 20, 22],
                [14, 22, 16],
                [2, 13, 15],
                [2, 15, 5],
                [3, 17, 21],
                [3, 21, 10],
            ]
        )
        vertex_attributes = jnp.tile(color, (len(vertices), 1))
        return Mesh(vertices, faces, vertex_attributes)

    @property
    def shape(self):
        return self.vertices.shape[:-1]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        self.current += 1
        if self.current <= len(self):
            return self[self.current - 1]
        raise StopIteration
