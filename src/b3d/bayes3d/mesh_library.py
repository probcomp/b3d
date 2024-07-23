import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class MeshLibrary:
    def __init__(self, vertices, faces, ranges, attributes, vertex_index_to_object):
        # cumulative (renderer inputs)
        self.vertices = vertices
        self.faces = faces
        self.ranges = ranges
        self.attributes = attributes
        self.vertex_index_to_object = vertex_index_to_object
        self.objects = []

    @staticmethod
    def make_empty_library():
        return MeshLibrary(
            jnp.empty((0, 3)),
            jnp.empty((0, 3), dtype=int),
            jnp.empty((0, 2), dtype=int),
            None,
            jnp.empty((0,), dtype=int),
        )

    def tree_flatten(self):
        return (
            (
                self.vertices,
                self.faces,
                self.ranges,
                self.attributes,
                self.vertex_index_to_object,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def get_object_name(self, obj_idx):
        return self.names[obj_idx]

    def get_num_objects(self):
        return len(self.ranges)

    def add_object(self, vertices, faces, attributes=None, name=None):
        """
        Given a new set of vertices and faces, update library.
        The input vertices/faces should correspond to a novel object, not a
        novel copy of an object already indexed by the library.
        """
        # if name is None:
        #     name = ""
        # self.names.append(name)
        self.objects.append((vertices, faces, attributes))

        current_length_of_vertices = len(self.vertices)
        current_length_of_faces = len(self.faces)
        current_length_of_ranges = len(self.ranges)

        self.vertices = jnp.concatenate((self.vertices, vertices))
        self.faces = jnp.concatenate((self.faces, faces + current_length_of_vertices))

        self.vertex_index_to_object = jnp.concatenate(
            (
                self.vertex_index_to_object,
                jnp.full(len(vertices), current_length_of_ranges),
            )
        )
        self.ranges = jnp.concatenate(
            (self.ranges, jnp.array([[current_length_of_faces, faces.shape[0]]]))
        )

        if attributes is not None:
            if self.attributes is None:
                self.attributes = attributes
            else:
                assert (
                    attributes.shape[0] == vertices.shape[0]
                ), "Attributes should be [num_vertices, num_attributes]"
                self.attributes = jnp.concatenate((self.attributes, attributes))

    def add_trimesh(self, mesh, vertex_scale_factor=1.0):
        vertices = jnp.array(mesh.vertices) * vertex_scale_factor
        vertices = vertices - jnp.mean(vertices, axis=0)
        faces = jnp.array(mesh.faces)
        vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
        self.add_object(vertices, faces, vertex_colors)
