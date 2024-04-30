import jax
import jax.numpy as jnp
import rerun as rr
import numpy as np
from b3d.renderer import Renderer
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Mesh:
    """
        vertices: (V, 2)
        faces: (F, 3)
        attributes: (V, A)
    """

    def __init__(self, vertices, faces, attributes):
        self.vertices = vertices
        self.faces = faces
        self.attributes = attributes

    @staticmethod
    def square_mesh(bottom_left, widthheight, attributes):
        """
        Create an axis-aligned square mesh, with a single attributes
        value.
        (This will be constructed as the union of 2 triangles with identical attributes.)
        Args:
            - bottom_left : jnp.array([x, y])
            - widthheight : jnp.array([width, height])
        """
        x, y = bottom_left
        width, height = widthheight
        vertices = jnp.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ])
        faces = jnp.array([
            [0, 1, 2],
            [0, 2, 3],
        ])
        return Mesh(vertices, faces, jnp.array([attributes, attributes, attributes, attributes]))

    @staticmethod
    def many_squares_meshes(bottom_lefts, widthheights, attributes):
        """
        Create multiple axis-aligned square meshes.
        Args:
            - bottom_lefts : jnp.array(N, 2)
            - widthheights : jnp.array(N, 2)
            - attributes : jnp.array(N, A)
        """
        vmapped_mesh = jax.vmap(Mesh.square_mesh, in_axes=(0, 0, 0))(bottom_lefts, widthheights, attributes)
        return Mesh.merge_vmapped(vmapped_mesh)

    def mesh_from_pixels(width, height, attributes):
        """
        Create a mesh from pixel values.
        Args:
            - width : int
            - height : int
            - attributes : jnp.array(width, height, A)
        """
        bottom_lefts = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
        bottom_lefts = jnp.stack(bottom_lefts, axis=-1).reshape(-1, 2)
        widthheights = jnp.ones((width * height, 2))
        attributes = attributes.reshape((-1, attributes.shape[-1]))
        return Mesh.many_squares_meshes(bottom_lefts, widthheights, attributes)

    @staticmethod
    def merge(mesh1, mesh2):
        """
        Merge two meshes.
        Args:
            - mesh1 : Mesh
            - mesh2 : Mesh
        """
        vertices = jnp.vstack([mesh1.vertices, mesh2.vertices])
        faces = jnp.vstack([mesh1.faces, mesh2.faces + len(mesh1.vertices)])
        attributes = jnp.vstack([mesh1.attributes, mesh2.attributes])
        return Mesh(vertices, faces, attributes)

    @staticmethod
    def merge_vmapped(vmapped_mesh):
        """
        Merge a vmapped mesh.
        Args:
            - vmapped_mesh : Mesh with a leading batch dimension.
        """
        vertices = vmapped_mesh.vertices.reshape((-1, 2))
        faces = vmapped_mesh.faces.reshape((-1, 3))
        attributes = vmapped_mesh.attributes.reshape((-1, vmapped_mesh.attributes.shape[-1]))
        
        n_vertices_per_face = vmapped_mesh.vertices.shape[1]
        N = faces.shape[0]
        indices = jnp.arange(N)
        shifts = (indices // n_vertices_per_face) * n_vertices_per_face
        faces_plus_offset = faces + jnp.array([shifts, shifts, shifts], dtype=int).transpose()

        return Mesh(vertices, faces_plus_offset, attributes)

    def scale(self, scale):
        """
        Scale the mesh by factor `scale`.
        Args:
            - scale : float
        """
        return Mesh(self.vertices * scale, self.faces, self.attributes)
    
    def translate(self, translation):
        """
        Translate the mesh by vector `translation`.
        Args:
            - translation : jnp.array([x, y])
        """
        return Mesh(self.vertices + translation, self.faces, self.attributes)
    
    def rotate(self, angle):
        """
        Rotate the mesh by angle `angle` (in radians) clockwise about the origin.
        Args:
            - angle : float
        """
        rotation_matrix = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        return Mesh(self.vertices @ rotation_matrix, self.faces, self.attributes)

    def transform_by_pose(self, pose):
        """
        Transform the mesh by pose `pose`.
        Args:
            - pose : jnp.array([x, y, angle])
        """
        return self.rotate(pose[2]).translate(pose[:2])

    def to_image(self, pose_2d, width, height, attributes_to_depth, default_attribute):
        """
        Get a width by height image of the 2D mesh at pose `pose_2d`.
        Args:
            - pose_2d : jnp.array([x, y, angle])
            - width : int
            - height : int
            - attributes_to_depth : function
                Function that maps attributes to depth.
            - default_attribute : The value returned at pixels where no triangle is visible.
        Returns:
            - Image: jnp.array(width, height, A) where A is the attribute dimension.
        """
        return rasterize_mesh(self, pose_2d, width, height, attributes_to_depth, default_attribute)

    ## Pytree registration ##
    def tree_flatten(self):
        return (
            (self.vertices, self.faces, self.attributes),
            None,
        )
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
## Rerun line segment visualization
def rerun_mesh_rep(mesh, attribute_to_color=(lambda rgbd: rgbd[:3])):
    # all_lines = jax.vmap(
    #     lambda i: lines_for_triangle(mesh, mesh.faces[i, :]),
    #     in_axes=(0,)
    # )(jnp.arange(mesh.faces.shape[0]))
    all_lines = [lines_for_triangle(mesh, i) for i in range(mesh.faces.shape[0])]
    # all_lines = jnp.array(all_lines)
    colors = jax.vmap(attribute_to_color, in_axes=(0,))(mesh.attributes[mesh.faces[:, 0], ...])
    return rr.LineStrips2D(all_lines, colors=colors)

def lines_for_triangle(mesh, face_index):
    """
    Get the line segments for a triangle.
    Args:
        - mesh : Mesh
        - face_index : int
    """
    face = mesh.faces[face_index]
    v0 = mesh.vertices[face[0]]
    v1 = mesh.vertices[face[1]]
    v2 = mesh.vertices[face[2]]
    return jnp.array([v0, v1, v2, v0])


## Rasterization
def rasterize_mesh(mesh, pose_2d, width, height, attributes_to_depth, default_attribute):
    # Set up renderer with long focal length.
    # We will put the 2D triangles in 3D space, at exactly
    # the focal distance, so we get a perfect 2D picture.
    # (We will actually move the triangles very slightly in the depth
    # dimension, to get a 2.5D effect where some triangles
    # can occlude others.)
    fx = width * height
    fy = fx
    cx, cy = -0.4999, -0.4999
    near, far = 0.1, (width * height)**2
    renderer = Renderer(
        width, height,
        fx, fy,
        cx, cy,
        near, far
    )

    # vertices to 3d, at depth fx
    depths = jax.vmap(attributes_to_depth)(mesh.attributes)
    vertices = jnp.concatenate((mesh.vertices, fx * jnp.ones((mesh.vertices.shape[0], 1))), axis=-1)
    # add on depths, scaled down a ton, so we get the right ordering of the triangles
    # when they overlap
    vertices += (jnp.array([0, 0, fx/1e5]).reshape(-1, 1) @ depths.reshape(1, -1)).transpose()
    
    # 3D pose as 4x4 matrix
    # with transform in xy plane given by pose_2d = [x, y, theta]
    x, y, theta = pose_2d
    pose_3d = jnp.array([
        [jnp.cos(-theta), -jnp.sin(-theta), 0, x],
        [jnp.sin(-theta), jnp.cos(-theta), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    rendered = renderer.render_attribute(
        pose_3d[None, ...],
        vertices,
        mesh.faces,
        jnp.array([0, mesh.faces.shape[0]]).reshape(1, 2), # 1 object, with all faces
        mesh.attributes
    )[0]

    image = jnp.where(
        (rendered[..., 3] == 0)[..., None],
        default_attribute,
        rendered[...]
    )

    return image