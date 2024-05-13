
import jax.numpy as jnp
import jax

def f(x):
    return jnp.linalg.norm(x)

grad = jax.jit(jax.grad(f))
grad(jnp.zeros(3))


import torch
x = torch.zeros(3, requires_grad=True)
y = torch.linalg.norm(x)
y.backward()
x.grad