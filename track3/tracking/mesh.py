import jax
import jax.numpy as jnp
import rerun as rr
import numpy as np
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Mesh:
    """
        vertices: (V, 2)
        faces: (F, 3)
        attributes: (V, A)
    """

    def __init__(self, vertices, faces, attributes):
        # if isinstance(faces, np.ndarray):
        #     assert not isinstance(faces, np.ndarray)
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

    ### Rendering ###
    def _get_in_triangle(self, point, face_index):
        """
        Check if a point is in a triangle.
        Args:
            - point : jnp.array([x, y])
            - face_index : int
        """
        face = self.faces[face_index]
        v0 = self.vertices[face[0]]
        v1 = self.vertices[face[1]]
        v2 = self.vertices[face[2]]
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = point - v0
        dot00 = jnp.dot(v0v2, v0v2)
        dot01 = jnp.dot(v0v2, v0v1)
        dot02 = jnp.dot(v0v2, v0p)
        dot11 = jnp.dot(v0v1, v0v1)
        dot12 = jnp.dot(v0v1, v0p)
        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom
        return (u >= 0) & (v >= 0) & (u + v < 1)

    def _get_in_triangle_array(self, point):
        return jax.vmap(lambda i : self._get_in_triangle(point, i))(jnp.arange(self.faces.shape[0]))        

    # OUTDATED: relied on attributes being per-face rather than per-file
    # def _get_depth_per_triangle(self, hits_triangle, attributes_to_depth):
    #     """
    #     Get the depth of each triangle.
    #     Args:
    #         - hits_triangle : jnp.array([F], bool)
    #             Boolean array indicating which triangles are hit.
    #         - attributes_to_depth : function
    #             Function that maps attributes to depth.
    #     """
    #     return jax.vmap(lambda i : jax.lax.cond(
    #         hits_triangle[i],
    #         lambda _: attributes_to_depth(self.attributes[i]),
    #         lambda _: jnp.inf,
    #         None
    #     ))(jnp.arange(self.faces.shape[0]))

    # OUTDATED: relied on attributes being per-face rather than per-file
    # def _get_pixel(self, i, j, attributes_to_depth, default_attribute):
    #     """
    #     Get the pixel value at pixel (i, j).
    #     The current implementation simply obtains the color
    #     at the center of this pixel [position (i+.5, j+.5)].
    #     Args:
    #         - i : int
    #         - j : int
    #         - attributes_to_depth : function
    #             Function that maps attributes to depth.
    #         - default_attribute : The value returned at pixels where no triangle is visible.
    #     """
    #     # (F,) boolean array
    #     hits_triangle = self._get_in_triangle_array(jnp.array([i + 0.501, j + 0.501]))
        
    #     # (F,) array: `depth` from `attributes_to_depth` for each visible
    #     # triangle; inf for each invisible triangle
    #     depth_per_triangle = self._get_depth_per_triangle(hits_triangle, attributes_to_depth)

    #     observed_triangle_index = jnp.argmin(depth_per_triangle)
    #     return jax.lax.select(
    #         hits_triangle[observed_triangle_index],
    #         self.attributes[observed_triangle_index],
    #         default_attribute
    #     )

    def to_image(self, width, height, attributes_to_depth, default_attribute):
        """
        Convert mesh to an image.
        Args:
            - width : int
            - height : int
            - attributes_to_depth : function
                Function that maps attributes to depth.
            - default_attribute : The value returned at pixels where no triangle is visible.
        """
        # get_pixel = lambda i, j: self._get_pixel(i, j, attributes_to_depth, default_attribute)
        # return jax.vmap(jax.vmap(get_pixel, in_axes=(0, None)), in_axes=(None, 0))(jnp.arange(width), jnp.arange(height))
        return rasterize_mesh(self, width, height, attributes_to_depth, default_attribute)

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
    colors = jax.vmap(attribute_to_color, in_axes=(0,))(mesh.attributes)
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
from b3d.renderer import Renderer
from b3d.pose import Pose
width = 64
height = 64
fx = width * height
fy = fx
cx = width/2
cy = height/2
near = 0.1
far = (width * height)**2
renderer = Renderer(
    width, height,
    fx, fy,
    cx, cy,
    near, far
)
def rasterize_mesh(mesh, width, height, attributes_to_depth, default_attribute):
    # vertices to 3d, at depth fx
    depths = jax.vmap(attributes_to_depth)(mesh.attributes)
    vertices = jnp.concatenate((mesh.vertices, fx * jnp.ones((mesh.vertices.shape[0], 1))), axis=-1)
    # add on depths, scaled down a ton, so we get the right ordering of the triangles
    # when they overlap
    vertices += (jnp.array([0, 0, 1/(100 * fx)]).reshape(-1, 1) @ depths.reshape(1, -1)).transpose()

    identity_pose_3d = Pose.identity() #.as_matrix()

    rendered = renderer.render_attribute(
        identity_pose_3d[None, ...],
        vertices,
        mesh.faces,
        jnp.array([0, mesh.faces.shape[0]]).reshape(1, 2), # 1 object, with all faces
        mesh.attributes
    )

    return rendered

    # _, _, triangle_ids, _ = renderer.rasterize(
    #     jnp.array(identity_pose_3d), # 1 object, at identity pose
    #     vertices,
    #     faces,
    #     jnp.array([0, faces.shape[0]]).reshape(1, 2), # 1 object, with all faces
    # )

    # measured_attributes = jax.lax.select(
    #     triangle_ids == 0,
    #     default_attribute,
    #     mesh.attributes[
    #         jnp.where(
    #             triangle_ids == 0,
    #             0,
    #             triangle_ids - 1)
    #         ]
    # )

    # return measured_attri/butes