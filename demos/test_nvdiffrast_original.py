from b3d.renderer_original import RendererOriginal
import jax.numpy as jnp
import b3d
import os
import trimesh
import time

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
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0


vertices = jnp.tile(vertices, (1000,1,1))
vertices_padded = b3d.pad_with_1(vertices)

num_timestep = 1000
sum_total = 0.0
start = time.time()
for _ in range(num_timestep):
    output, = renderer.rasterize_original(
        vertices_padded, faces, resolution
    )
    sum_total += output.sum()
end = time.time()
print(sum_total)
print(f"FPS: {num_timestep/(end-start)}")


output, = renderer.rasterize_original(
    vertices, faces, resolution
)
for i in range(len(output)):
    print(output[i].sum())




vertices = jnp.tile(jnp.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.0, 1.0],
]
)[None,...], (5,1,1))
faces = jnp.array([[0,1,2]], dtype=jnp.int32)

output, = renderer.rasterize_original(
    vertices, faces, resolution
)
for i in range(len(output)):
    print(output[i].sum())



# glctx = dr.RasterizeGLContext()

# import torch
# import numpy as np
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vertices = jnp.tile(jnp.array([
#         [0.0, 0.0, 0.0, 1.0],
#         [0.4, 0.0, 0.0, 1.0],
#         [0.0, 0.4, 0.0, 1.0],
# ]
# )[None,...], (3,1,1))
# faces = jnp.array([[0,1,2]], dtype=jnp.int32)

# vertices_torch = torch.tensor(np.array(vertices), device=device)
# faces_torch = torch.tensor(np.array(faces), device=device,dtype=torch.int32)
# rast_out, _ = dr.rasterize(glctx, vertices_torch, faces_torch, resolution=[height, width])
# for i in range(len(rast_out)):
#     print(rast_out[i].sum())




# vertices = jnp.tile(jnp.array([
#         [0.0, 0.0, 0.0, 1.0],
#         [0.2, 0.0, 0.0, 1.0],
#         [0.0, 0.2, 0.0, 1.0],
# ]
# )[None,...], (5,1,1))
# vertices_torch = torch.tensor(np.array(vertices), device=device)
# faces_torch = torch.tensor(np.array(faces), device=device,dtype=torch.int32)
# rast_out, _ = dr.rasterize(glctx, vertices_torch, faces_torch, resolution=[height, width])
# for i in range(len(rast_out)):
#     print(rast_out[i].sum())





# b3d.get_rgb_pil_image(output[1,..., :3]).save("0.png")



# mesh_path = os.path.join(
#     b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
# )
# mesh = trimesh.load(mesh_path)
# vertices = jnp.array(mesh.vertices)
# vertices = vertices - jnp.mean(vertices, axis=0)
# faces = jnp.array(mesh.faces)
# vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
# vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0


# vertices_tiled = jnp.tile(vertices[None,...], (30, 1,1) ) + jnp.array([0.0, 0.0, 0.2])
# vertices_tiled_pad_1 = b3d.pad_with_1(vertices_tiled)
# vertices_projected = vertices_tiled_pad_1 @ renderer.projection_matrix_t

# output, = renderer.rasterize_original(
#     vertices_projected, faces, jnp.array([[0, len(faces)]]), resolution
# )
# for i in range(len(output)):
#     print(output[i].sum())


# print(output[0].sum())
# print(output[1].sum())


# vertices_tiled = jnp.tile(vertices[None,...], (10, 1,1) ) + jnp.linspace(jnp.array([0.0, 0.0, 0.2]), jnp.array([0.0, 0.0, 0.4]), 10)[:,None]
# output, = renderer.rasterize(
#     vertices_tiled, faces
# )


# b3d.get_rgb_pil_image(output[1,..., :3]).save("0.png")



# vertices_tiled =  jnp.tile(vertices[None,...], (1, 1,1) )

# output, = renderer.rasterize_original(
#     vertices_projected, faces, jnp.array([[0, 0]]), resolution
# )

# print(output[0].sum())
# print(output[-1].sum())
# b3d.get_rgb_pil_image(output[-1,...,:3]).save("1.png")


# glctx = dr.RasterizeGLContext()

# import torch
# import numpy as np
# vertices_tiled_projected = b3d.pad_with_1(vertices_tiled) @ renderer.projection_matrix_t
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vertices_torch = torch.tensor(np.array(vertices_tiled_projected), device=device)
# faces_torch = torch.tensor(np.array(faces), device=device,dtype=torch.int32)


# rast_out, _ = dr.rasterize(glctx, vertices_torch, faces_torch, resolution=[height, width])
# print(rast_out[0].sum())
# print(rast_out[1].sum())



# # import rerun as rr
# # rr.init("demo")
# # rr.connect("127.0.0.1:8812")
# # rr.log("torch", rr.Image(output[0,...,:3]))


