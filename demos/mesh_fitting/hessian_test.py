import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.differentiable_renderer
import b3d.tessellation as t
import b3d.utils as u
import os
import rerun as rr
import optax
from tqdm import tqdm

import demos.mesh_fitting.model as m

@jax.custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)

jax.jacrev(jax.jacrev(f))(jnp.ones(3), jnp.ones(3))