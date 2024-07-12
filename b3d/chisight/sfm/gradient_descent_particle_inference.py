import jax
import jax.numpy as jnp
import optax
from b3d.utils import keysplit
from b3d.pose import Pose, Rot
from b3d.camera import camera_from_screen_and_depth
from .utils import reprojection_error


def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict.'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn
label_fn = map_nested_fn(lambda k, _: k)


def map_over_nested_dict_values(f):
  '''Recursively apply `f` to the values of a nested dict.'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else f(v))
            for k, v in nested_dict.items()}
  return map_fn


def init_params(key, uvs0, cam0, intr):
    _, key = keysplit(key,1,1)

    # Initialize 3d keypoints in 
    # fixed camera frame
    N = uvs0.shape[0]
    z = jax.random.normal(key, (N,))*.1 + 6.
    xs = cam0(camera_from_screen_and_depth(uvs0, z, intr))
    params = {"xs": xs}
    
    return params

def get_particle_positions(params):
    return params["xs"]


def loss_function(params, uvs0, uvs1, cam0, cam1, intr):                             
    xs = get_particle_positions(params)
    err0 = reprojection_error(xs, uvs0, cam0, intr)    
    err1 = reprojection_error(xs, uvs1, cam1, intr)    
    return (
        jnp.mean(err0 + err1)
    )

loss_func_grad = jax.value_and_grad(loss_function, argnums=(0,))



def make_fit(key, uvs0, uvs1, cam0, cam1, intr, learning_rate=1e-3):

    optimizer = optax.multi_transform(
        {
            'xs': optax.adam(learning_rate),
        },
        label_fn
    )

    @jax.jit
    def step(carry, _):
        params, opt_state, loss_args = carry
        ell, (grads,) = loss_func_grad(params, *loss_args)
        updates, opt_state = optimizer.update(grads, opt_state)
        updates = map_over_nested_dict_values(jnp.nan_to_num)(updates)
        params = optax.apply_updates(params, updates)
        params['xs'] = params['xs'].at[:,2].set(jnp.clip(params['xs'][:,2], 0., jnp.inf))
        return ((params, opt_state, loss_args), ell)


    params = init_params(key, uvs0, cam0, intr)
    loss_args = (uvs0, uvs1, cam0, cam1, intr)

    def fit(params, steps=1_000):
        _, subkey = keysplit(key,1,1)
        opt_state = optimizer.init(params)
        loss_args = (uvs0, uvs1, cam0, cam1, intr)
        (params, opt_state, loss_args), losses = jax.lax.scan(step, (params, opt_state, loss_args), xs=None, length=steps)

        return params, losses

    return params, fit

