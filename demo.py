import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import jax_gl_renderer

image_width, image_height, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = jax_gl_renderer.JaxGLRenderer(w, image_height, fx,fy, cx,cy, near, far)

box_mesh = trimesh.creation.box()
vertices = box_mesh.vertices
faces = box_mesh.faces
ranges = jnp.array([[0, len(faces)],[0, len(faces)]])

pose = jnp.array([
    [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 3.],
        [0., 0., 0., 1.],
    ],
    [
        [1., 0., 0., 1.],
        [0., 1., 0., 0.],
        [0., 0., 1., 4.],
        [0., 0., 0., 1.],
    ], 
])

uvs, object_ids, triangle_ids = jax_renderer.render_to_barycentrics(pose, vertices, faces, ranges)

fig, axs = plt.subplots(1,3)
axs[0].imshow(uvs[...,0])
axs[1].imshow(object_ids)
axs[2].imshow(triangle_ids)
fig.savefig("demo.png")