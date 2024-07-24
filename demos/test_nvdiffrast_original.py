import os
import time

import b3d
import jax
import jax.numpy as jnp
import trimesh
from b3d.renderer_original import RendererOriginal

width = 200
height = 100
fx = 100.0
fy = 100.0
cx = 100.0
cy = 50.0
near = 0.001
far = 16.0
renderer = RendererOriginal(width, height, fx, fy, cx, cy, near, far)
resolution = jnp.array([height, width]).astype(jnp.int32)

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0) + jnp.array([0.0, 0.0, 0.2])
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0


rasterize_jit = jax.jit(renderer.rasterize)

N = 10
vertices_tiled = (
    jnp.tile(vertices[None, ...], (N, 1, 1))
    + jnp.linspace(jnp.array([0.0, 0.0, 0.05]), jnp.array([0.0, 0.0, 0.4]), N)[:, None]
)
(output,) = rasterize_jit(vertices_tiled, faces)
for i in range(len(output)):
    print(output[i].sum())
print("====")
N = 20
vertices_tiled = (
    jnp.tile(vertices[None, ...], (N, 1, 1))
    + jnp.linspace(jnp.array([0.0, 0.0, 0.05]), jnp.array([0.0, 0.0, 0.4]), N)[:, None]
)
(output,) = rasterize_jit(vertices_tiled, faces)
for i in range(len(output)):
    print(output[i].sum())


N = 1000
vertices_tiled = (
    jnp.tile(vertices[:100], (N, 1, 1))
    + jnp.linspace(jnp.array([0.0, 0.0, 0.05]), jnp.array([0.0, 0.0, 0.4]), N)[:, None]
)

num_timestep = 1000
sum_total = 0.0
start = time.time()
for _ in range(num_timestep):
    (output,) = rasterize_jit(vertices_tiled, faces)
    sum_total += output.sum()
end = time.time()
print(sum_total)
print(f"FPS: {num_timestep/(end-start)}")
b3d.get_rgb_pil_image(output[99, ..., :3]).save("0.png")


N = 1000
vertices = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)

faces = jnp.array(
    [
        [0, 1, 2],
    ]
)
num_timestep = 1000
sum_total = 0.0
start = time.time()
for _ in range(num_timestep):
    (output,) = rasterize_jit(vertices[None, ...], faces)
    sum_total += output.sum()
end = time.time()
print(sum_total)
print(f"FPS: {num_timestep/(end-start)}")
