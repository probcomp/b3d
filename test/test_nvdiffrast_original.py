# from b3d.renderer_original import Renderer
# import jax.numpy as jnp
# import b3d.nvdiffrast_original.torch as dr

# renderer = Renderer(100,159, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0)

# vertices = jnp.array([
#         [0.0, 0.0, 0.0, 1.0],
#         [1.0, 0.0, 0.0, 1.0],
#         [0.0, 1.0, 0.0, 1.0],
# ]
# )[None,...]
# faces = jnp.array([[0,1,2]], dtype=jnp.int32)
# vertex_colors = jnp.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]])
# resolution = jnp.array([100, 150]).astype(jnp.int32)

# output, = renderer.rasterize(
#     vertices, faces, jnp.array([[0, len(faces)]]), resolution
# )
# print(output.sum())

# print("Torch")
# resolutions = [1024, 512, 256, 128, 64, 32]
# for resolution in resolutions:
#     rast_out, _ = dr.rasterize(glctx, vertices[None,...], faces, resolution=[resolution, resolution])
#     color   , _ = dr.interpolate(vertex_colors, rast_out, faces)
#     rr.log("torch", rr.Image(color.cpu().numpy()[0]))

#     num_timestep = 1000
#     start = time.time()
#     for _ in range(num_timestep):
#         rast_out, _ = dr.rasterize(glctx, vertices[None,...], faces, resolution=[resolution, resolution])
#         print(rast_out.sum())
#     end = time.time()

#     print(f"Resolution: {resolution}x{resolution}, FPS: {num_timestep/(end-start)}")



# import rerun as rr
# rr.init("demo")
# rr.connect("127.0.0.1:8812")
# rr.log("torch", rr.Image(output[0,...,:3]))
