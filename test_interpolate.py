import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import jax_gl_renderer
from jax.scipy.spatial.transform import Rotation as Rot
from jax_gl_renderer import Pose
import rerun as rr

rr.init("demo.py")
rr.connect("127.0.0.1:8812")

width=100
height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=6.0
renderer = jax_gl_renderer.JaxGLRenderer(
    width, height, fx, fy, cx, cy, near, far
)

attributes = jnp.array([[10.0, 10.0, 10.0, 10.0],[20.0, 20.0, 20.0, 20.0],[30.0, 30.0, 30.0, 30.0],])
uvs = jnp.array([0.5, 0.5])[None, None, None, ...]
triangle_ids = jnp.array([1])[None, None,...]
faces = jnp.array([[0, 1, 2]])

for i in (attributes, uvs, triangle_ids, faces):
    print(i.shape, i.dtype)

image = renderer.interpolate(
    attributes, uvs, triangle_ids, faces
)