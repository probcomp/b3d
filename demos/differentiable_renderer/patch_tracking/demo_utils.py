import jax.numpy as jnp
import jax
import os
import trimesh
import b3d
from b3d import Pose
import b3d.patch_tracking.utils as ptutils

### Utils ###

# Define pose math libraries
def rotation_from_axis_angle(axis, angle):
    """Creates a rotation matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The rotation matrix. Shape (3, 3)
    """
    sina = jnp.sin(angle)
    cosa = jnp.cos(angle)
    direction = axis / jnp.linalg.norm(axis)
    # rotation matrix around unit vector
    R = jnp.diag(jnp.array([cosa, cosa, cosa]))
    R = R + jnp.outer(direction, direction) * (1.0 - cosa)
    direction = direction * sina
    R = R + jnp.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return R

def transform_from_rot(rotation):
    """Creates a pose matrix from a rotation matrix.

    Args:
        rotation (jnp.ndarray): The rotation matrix. Shape (3, 3)
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return jnp.vstack(
        [jnp.hstack([rotation, jnp.zeros((3, 1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_axis_angle(axis, angle):
    """Creates a pose matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return transform_from_rot(rotation_from_axis_angle(axis, angle))

# calculate sequence of pose transformations
r_mat = transform_from_axis_angle(jnp.array([0,0,1]), jnp.pi/2)
vec_transform_axis_angle = jax.vmap(transform_from_axis_angle, (None, 0))

### Convenience wrapper for common code used in demos ###
def get_renderer_boxdata_and_patch():
    width = 100
    height = 128
    fx=64.0
    fy=64.0
    cx=64.0
    cy=64.0
    near=0.001
    far=16.0

    renderer = b3d.Renderer(
        width, height, fx, fy, cx, cy, near, far
    )

    # Rotating box data
    mesh_path = os.path.join(b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
    mesh = trimesh.load(mesh_path)
    cheezit_object_library = b3d.MeshLibrary.make_empty_library()
    cheezit_object_library.add_trimesh(mesh)
    rots = vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
    in_place_rots = b3d.Pose.from_matrix(rots)
    cam_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.15, 0.15, 0.0]),
        jnp.array([0.0, 0.0, 0.0])
    )
    X_WC = cam_pose
    compound_pose = X_WC.inv() @ in_place_rots
    rgbs, depths = renderer.render_attribute_many(
        compound_pose[:,None,...],
        cheezit_object_library.vertices,
        cheezit_object_library.faces,
        jnp.array([[0, len(cheezit_object_library.faces)]]),
        cheezit_object_library.attributes
    )
    observed_rgbds = jnp.concatenate([rgbs, depths[...,None]], axis=-1)
    xyzs_C = unproject_depth_vec(depths, renderer)
    xyzs_W = X_WC.apply(xyzs_C)
    box_data = (observed_rgbds, rots)

    # Get patch
    center_y, center_x = 45, 40
    del_pix = 5
    patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
    patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
    patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        patch_points_C, patch_rgbs, patch_points_C[:,2] / fx
    )
    X_CP = Pose.from_translation(patch_vertices_C.mean(0))
    patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
    patch_vertices_W = X_WC.apply(patch_vertices_C)
    X_WP = X_WC @ X_CP
    patch_data = ((patch_vertices_P, patch_faces, patch_vertex_colors), X_WP)

    return (renderer, box_data, patch_data, X_WC)

def get_rotating_box_data(renderer):
    mesh_path = os.path.join(b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
    mesh = trimesh.load(mesh_path)
    cheezit_object_library = b3d.MeshLibrary.make_empty_library()
    cheezit_object_library.add_trimesh(mesh)
    rots = vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
    in_place_rots = b3d.Pose.from_matrix(rots)
    cam_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.15, 0.15, 0.0]),
        jnp.array([0.0, 0.0, 0.0])
    )
    X_WC = cam_pose
    compound_pose = X_WC.inv() @ in_place_rots
    rgbs, depths = renderer.render_attribute_many(
        compound_pose[:,None,...],
        cheezit_object_library.vertices,
        cheezit_object_library.faces,
        jnp.array([[0, len(cheezit_object_library.faces)]]),
        cheezit_object_library.attributes
    )
    observed_rgbds = jnp.concatenate([rgbs, depths[...,None]], axis=-1)
    xyzs_C = ptutils.unproject_depth_vec(depths, renderer)
    xyzs_W = X_WC.apply(xyzs_C)
    
    return (X_WC, rgbs, xyzs_W, observed_rgbds)