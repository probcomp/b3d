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
import nvdiffrast.torch as dr
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
faces = torch.tensor(np.array(faces), device=device, dtype=torch.int32)
vertex_colors = torch.tensor(np.array(vertex_colors), device=device)

glctx = dr.RasterizeGLContext()  # if use_opengl else dr.RasterizeCudaContext()

resolution = 1024

convert_to_torch = lambda x: torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack((x)))
convert_to_jax = lambda x: jax.dlpack.from_dlpack(
    torch.utils.dlpack.to_dlpack((x.contiguous()))
)

import jax

vertices_jax_4 = jnp.array(vertices.cpu().numpy())
vertices_jax = jnp.array(vertices.cpu().numpy())[..., :3]
faces_jax = jnp.array(faces.cpu().numpy())
vertex_colors_jax = jnp.array(vertex_colors.cpu().numpy())
ranges_jax = jnp.array([[0, len(faces_jax)]])


def render(vertices_jax_4, faces_jax):
    rast_out, _ = dr.rasterize(
        glctx,
        convert_to_torch(vertices_jax_4)[None, ...],
        convert_to_torch(faces_jax),
        resolution=[resolution, resolution],
    )
    return convert_to_jax(rast_out)


render(vertices_jax_4, faces_jax)
