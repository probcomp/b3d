import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import imageio
import b3d
from tqdm import tqdm
import jax.numpy as jnp
import b3d.nvdiffrast_original.torch as dr
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import trimesh

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0

vertices = torch.tensor(np.array(vertices), device=device)
vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
faces = torch.tensor(np.array(faces), device=device,dtype=torch.int32)
vertex_colors = torch.tensor(np.array(vertex_colors), device=device)


# vertices = torch.tensor(
#     [
#         [0.0, 0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0, 1.0],
#     ],device=device
# )

# faces = torch.tensor(
#     [
#         [0, 1, 2],
#     ],device=device,dtype=torch.int32
# )

# vertex_colors = torch.tensor(
#     [
#         [1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#         [0.0, 0.0, 1.0],
#     ],device=device
# )


glctx = dr.RasterizeGLContext() #if use_opengl else dr.RasterizeCudaContext()

import rerun as rr
rr.init("demo")
rr.connect("127.0.0.1:8812")



print("Torch")
resolutions = [1024, 512, 256, 128, 64, 32]
for resolution in resolutions:
    rast_out, _ = dr.rasterize(glctx, vertices[None,...], faces, resolution=[resolution, resolution])
    color   , _ = dr.interpolate(vertex_colors, rast_out, faces)
    rr.log("torch", rr.Image(color.cpu().numpy()[0]))
    sum = 0
    num_timestep = 1000
    start = time.time()
    for _ in range(num_timestep):
        rast_out, _ = dr.rasterize(glctx, vertices[None,...], faces, resolution=[resolution, resolution])
        sum += rast_out.sum()
    end = time.time()
    print(sum)

    print(f"Resolution: {resolution}x{resolution}, FPS: {num_timestep/(end-start)}")


vertices_jax_4 = jnp.array(vertices.cpu().numpy())
vertices_jax = jnp.array(vertices.cpu().numpy())[...,:3]
faces_jax = jnp.array(faces.cpu().numpy())
vertex_colors_jax = jnp.array(vertex_colors.cpu().numpy())
ranges_jax = jnp.array([[0, len(faces_jax)]])
poses = b3d.Pose.from_translation(jnp.array([0.0, 0.0, 5.1]))[None, None, ...]

import jax
print("JAX NVdiffrast Original")
from b3d.renderer_original import Renderer as RendererOriginal

for resolution in resolutions:
    renderer = RendererOriginal(resolution, resolution, 100.0, 100.0, resolution/2.0, resolution/2.0, 0.01, 10.0, num_layers=1)
    render_jit = jax.jit(renderer.rasterize)
    num_timestep = 1000
    resolution_array = jnp.array([resolution, resolution]).astype(jnp.int32)
    sum = 0 
    start = time.time()
    for _ in range(num_timestep):
        output, = render_jit(
            vertices_jax_4[None,...], faces_jax, ranges_jax, resolution_array
        )
        sum += output.sum()
    end = time.time()
    print(sum)

    print(f"Resolution: {resolution}x{resolution}, FPS: {num_timestep/(end-start)}")



import jax

print("JAX")
for resolution in resolutions:
    renderer = b3d.Renderer(resolution, resolution, 100.0, 100.0, resolution/2.0, resolution/2.0, 0.01, 10.0, num_layers=1)
    render_jit = jax.jit(renderer.render_attribute_many)
    image,_ = render_jit(poses, vertices_jax, faces_jax, ranges_jax, vertex_colors_jax)
    rr.log("jax", rr.Image(image[0]))

    num_timestep = 1000
    start = time.time()
    for _ in range(num_timestep):
        image = render_jit(poses, vertices_jax, faces_jax, ranges_jax, vertex_colors_jax)
    end = time.time()

    print(f"Resolution: {resolution}x{resolution}, FPS: {num_timestep/(end-start)}")


convert_to_torch = lambda x: torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack((x)))

print("JAX through torch DLPACK")
for resolution in resolutions:

    rast_out, _ = dr.rasterize(glctx, convert_to_torch(vertices_jax_4)[None,...], convert_to_torch(faces_jax), resolution=[resolution, resolution])
    color   , _ = dr.interpolate(convert_to_torch(vertex_colors_jax), rast_out, convert_to_torch(faces_jax))
    rr.log("torch", rr.Image(color.cpu().numpy()[0]))

    num_timestep = 1000
    start = time.time()
    for _ in range(num_timestep):
        rast_out, _ = dr.rasterize(glctx, convert_to_torch(vertices_jax_4)[None,...], convert_to_torch(faces_jax), resolution=[resolution, resolution])
        color   , _ = dr.interpolate(convert_to_torch(vertex_colors_jax), rast_out, convert_to_torch(faces_jax))
    end = time.time()

    print(f"Resolution: {resolution}x{resolution}, FPS: {num_timestep/(end-start)}")


