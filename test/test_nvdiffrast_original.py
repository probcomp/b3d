from b3d.renderer_original import RendererOriginal
import b3d.nvdiffrast_original.torch as dr
import jax.numpy as jnp
import b3d
import os
import trimesh

width = 200
height = 100
fx = 200.0
fy = 200.0
cx = 100.0
cy = 50.0
near = 0.001
far = 16.0
renderer = RendererOriginal(width, height, fx, fy, cx, cy, near, far)

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0

vertices_tiled =  jnp.tile(vertices[None,...], (1000, 1,1) )+ jnp.array([0.0, 0.0, 0.3])
output, = renderer.rasterize(
    vertices_tiled, faces
)
print(output[0].sum())
print(output[-1].sum())
b3d.get_rgb_pil_image(output[0,...,:3]).save("1.png")


glctx = dr.RasterizeGLContext()

import torch
import numpy as np
vertices_tiled_projected = b3d.pad_with_1(vertices_tiled) @ renderer.projection_matrix_t
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vertices_torch = torch.tensor(np.array(vertices_tiled_projected), device=device)
faces_torch = torch.tensor(np.array(faces), device=device,dtype=torch.int32)


rast_out, _ = dr.rasterize(glctx, vertices_torch, faces_torch, resolution=[height, width])
print(rast_out[0].sum())
print(rast_out[1].sum())



# import rerun as rr
# rr.init("demo")
# rr.connect("127.0.0.1:8812")
# rr.log("torch", rr.Image(output[0,...,:3]))


