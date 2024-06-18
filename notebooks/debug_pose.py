import jax.numpy as jnp
from b3d import Pose
import jax

positions = jnp.linspace(jnp.array([0,0,0]), jnp.array([0,0,5]), 5)
quats = jnp.linspace(jnp.array([0,0,0,1]), jnp.array([0,0,0,1]), 5)

test_poses = Pose(positions, quats)

print(len(test_poses))

def sum(poses):
    sum = jnp.zeros(3)
    for p in poses:
        sum = sum.at[0:3].add(p.pos[2])
    return sum

print(sum(test_poses))

sum_jit = jax.jit(sum)

print(sum_jit(test_poses))