import unittest
import genjax
from genjax import Pytree
import jax
import jax.numpy as jnp
import numpy as np
from b3d import Pose
import b3d
from b3d.chisight.shared.particle_system import (
    get_sparse_test_model_and_args, 
    get_cameras, 
    set_camera_choice, 
    get_observations, 
    set_sensor_coordinates_choice
)

# Get a minimal model for testing
key = jax.random.PRNGKey(np.random.randint(1_000))
T = 4
N = 5
K = 3
model, args = get_sparse_test_model_and_args(T, N, K)

class MeshTests(unittest.TestCase):
    
    def test_camera_setter(self):
        global key;
        global model, args;
        for t in range(T):
            key,_ = jax.random.split(key)

            ch = set_camera_choice(t, Pose.id())
            tr, w = model.importance(key, ch, args)
            cams = get_cameras(tr)
            
            assert jnp.allclose(cams[t].pos, jnp.zeros(3))


    def test_observation_setter(self):
        global key;
        global model, args;
        for t in range(T):
            key,_ = jax.random.split(key)
            
            ch = set_sensor_coordinates_choice(t, jnp.zeros((N,2)))
            tr, w = model.importance(key, ch, args)
            uvs = get_observations(tr)

            # TODO: Wait for vmap importance bug has been resolved
            assert jnp.allclose(uvs[t], jnp.zeros((N,2)))