import jax.numpy as jnp

def dummy_solver(task_input):
    def grid_solver(partition):
        return jnp.ones_like(partition) / len(partition)

    return {
        "laplace": {},
        "grid": grid_solver,
        "mala": {},
        "multi_initialized_mala": {}
    }