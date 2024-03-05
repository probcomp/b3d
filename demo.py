import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh
from jax_gl_renderer import JaxGLRenderer, projection_matrix_from_intrinsics, get_assets_dir

meshes = []
path = os.path.join(get_assets_dir(), "cube.obj")
mesh = trimesh.load(path)
mesh.vertices  = mesh.vertices * jnp.array([1.0, 1.0, 1.0]) * 0.7
meshes.append(mesh)

path = os.path.join(get_assets_dir(), "bunny.obj")
bunny_mesh = trimesh.load(path)
bunny_mesh.vertices  = bunny_mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
meshes.append(bunny_mesh)

w,h, fx,fy, cx,cy, near, far = 200, 100, 200.0, 200.0, 100.0, 50.0, 0.001, 16.0
jax_renderer = JaxGLRenderer(w,h, fx,fy, cx,cy, near, far)
