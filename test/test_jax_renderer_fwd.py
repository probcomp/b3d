import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh
from jax_gl_renderer import JaxGLRenderer, projection_matrix_from_intrinsics, get_assets_dir

import viser
server = viser.ViserServer()


w,h, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = JaxGLRenderer(w,h, fx,fy, cx,cy, near, far)

meshes = []


path = os.path.join(get_assets_dir(), "cube.obj")
mesh = trimesh.load(path)
mesh.vertices  = mesh.vertices * jnp.array([1.0, 1.0, 1.0]) * 0.7
meshes.append(mesh)

path = os.path.join(get_assets_dir(), "bunny.obj")
bunny_mesh = trimesh.load(path)
bunny_mesh.vertices  = bunny_mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
meshes.append(bunny_mesh)


all_vertices = [jnp.array(mesh.vertices) for mesh in meshes]
all_faces = [jnp.array(mesh.faces) for mesh in meshes]
vertices_lens = jnp.array([len(verts) for verts in all_vertices])
vertices_lens_cumsum = jnp.pad(jnp.cumsum(vertices_lens),(1,0))
faces_lens = jnp.array([len(faces) for faces in all_faces])
faces_lens_cumsum = jnp.pad(jnp.cumsum(faces_lens),(1,0))

vertices = jnp.concatenate(all_vertices, axis=0)
vertices = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
faces = jnp.concatenate([faces + vertices_lens_cumsum[i] for (i,faces) in enumerate(all_faces)])

object_indices = jnp.array([1, 0])
ranges = jnp.hstack([faces_lens_cumsum[object_indices].reshape(-1,1), faces_lens[object_indices].reshape(-1,1)])

poses = jnp.tile(jnp.eye(4).reshape(1,4,4), (4000,1,1))
poses = poses.at[:, 2,3].set(5.0)

poses = poses.at[:, 1,3].set(jnp.linspace(-0.2, 0.5, len(poses)))
poses2 = poses.at[:, 1,3].set(jnp.linspace(-0.0, 1.5, len(poses)))
poses = jnp.stack([poses, poses2], axis=1)

render_jit = jax.jit(jax_renderer.render)

image = render_jit(
    poses,
    vertices,
    faces,
    ranges,
)

server.reset_scene()

server.add_point_cloud(
    "image1",
    points=np.array(image[20]).reshape(-1,3),
    colors=np.array([1.0, 0.0, 0.0]),
    point_size=0.01
)

from IPython import embed; embed()