import genjax
import jax
import jax.numpy as jnp
import b3d
import trimesh
import os
import rerun as rr

import importlib
importlib.reload(b3d)
import b3d

b3d.rr_init()

renderer = b3d.RendererOriginal()
object_poses = [b3d.Pose.from_position_and_target(jnp.array([0.3, 0.0, 0.1]), jnp.zeros(3)).inv()]
delta_pose = b3d.Pose.sample_gaussian_vmf_pose(jax.random.PRNGKey(0), b3d.Pose.identity(), 0.001, 10000.0)
for i in range(100):
    object_poses.append(object_poses[-1] @ delta_pose)
object_poses_stacked = b3d.Pose.stack_poses(object_poses)

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices, faces, colors = b3d.get_vertices_faces_colors_from_mesh(mesh)

images = renderer.render_rgbd_many(object_poses_stacked[:,None].apply(vertices), faces, jnp.tile(colors, (len(object_poses_stacked), 1 , 1)))

xyzs = jnp.vectorize(b3d.xyz_from_depth, excluded=(1,2,3,4,), signature='(h,w)->(h,w,3)')(images[...,3], renderer.fx, renderer.fy, renderer.cx, renderer.cy)

for i in range(len(images)):
    rr.set_time_sequence("frame", i)
    rr.log("/image", rr.Image(images[i,...,:3]))
    rr.log("/point_cloud", rr.Points3D(xyzs[i].reshape(-1,3), colors=images[i,...,:3].reshape(-1,3)))

i,j = 69,118
def get_mesh_from_pixel_index(i,j, xyz, image, fx, fy):
    window = 5
    
    patch_points = jax.lax.dynamic_slice(xyz, (i-window,j-window,0), (2*window+1,2*window+1,3)).reshape(-1,3)
    patch_point_colors = jax.lax.dynamic_slice(image, (i-window,j-window,0), (2*window+1,2*window+1,3)).reshape(-1,3)
    
    mesh = b3d.make_mesh_from_point_cloud_and_resolution(
        patch_points, patch_point_colors, patch_points[:,2][...,None] * jnp.array([1.0 / fx, 1.0 / fy , 0.0]) + jnp.array([0.0, 0.0, 0.0005])
    )
    return mesh[:3]
get_mesh_from_pixel_index(i,j, xyzs[0], images[0], renderer.fx, renderer.fy)

ijs = jax.random.choice(jax.random.PRNGKey(10), jnp.stack(jnp.where(xyzs[0,...,2] > 0)).transpose(), shape =(60,))
meshes = jax.vmap(get_mesh_from_pixel_index, in_axes=(0,0,None,None,None,None))(ijs[:,0], ijs[:,1], xyzs[0], images[0], renderer.fx, renderer.fy)

rr.set_time_sequence("frame", 0)
for i in range(len(meshes[0])):
    rr.log(f"/patch/{i}", rr.Mesh3D(vertex_positions=meshes[0][i], indices=meshes[1][i], vertex_colors=meshes[2][i]))

vertices, faces, vertex_colors = meshes
def merge_meshes(vertices, faces, vertex_colors):
    num_meshes, num_vertices, _ = vertices.shape
    merged_vertices = jnp.concatenate(vertices, axis=0)
    merged_faces = jnp.concatenate(faces + (jnp.arange(num_meshes) * num_vertices)[:,None,None], axis=0)
    merged_vertex_colors = jnp.concatenate(vertex_colors, axis=0)
    return merged_vertices, merged_faces, merged_vertex_colors

merged_mesh = merge_meshes(vertices, faces, vertex_colors)
rr.log("merged_mesh", rr.Mesh3D(vertex_positions=merged_mesh[0], indices=merged_mesh[1], vertex_colors=merged_mesh[2]))

rerendered_image = renderer.render_rgbd(merged_mesh[0], merged_mesh[1], merged_mesh[2])
rr.log("/image/rerender", rr.Image(rerendered_image[...,:3]))
    