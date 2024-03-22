import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh
from jax_gl_renderer import JaxGLRenderer, projection_matrix_from_intrinsics, get_assets_dir

w,h, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = JaxGLRenderer(w,h, fx,fy, cx,cy, near, far)

box_mesh = trimesh.creation.box()
vertices = box_mesh.vertices
faces = box_mesh.faces
ranges = jnp.array([[0, len(faces)]])
pose = jnp.eye(4)[None,...]

jax_renderer.render_to_barycentrics(pose, vertices, faces, ranges)
