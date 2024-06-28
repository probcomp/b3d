import genjax
from genjax import Pytree
import jax
from b3d import Pose
import b3d
import b3d.chisight.shared.particle_system as particle_system


key = jax.random.PRNGKey(125)

T = Pytree.const(4)
N = Pytree.const(5)
K = Pytree.const(3)

particle_prior_params = (Pose.identity(), .5, 0.25)
object_prior_params   = (Pose.identity(), 2., 0.5)
camera_prior_params = (Pose.identity(), 0.1, 0.1)
instrinsics = Pytree.const(b3d.camera.Intrinsics(120, 100, 50., 50., 50., 50., 0.001, 16.))
sigma_obs = 0.2


model = particle_system.sparse_gps_model
latent_args = (
        T, # const object
        N, # const object
        K, # const object
        particle_prior_params,
        object_prior_params,
        camera_prior_params
)
observation_args = (instrinsics, sigma_obs)
args = (latent_args, observation_args)
jsimulate = jax.jit(model.simulate)
jimportance = jax.jit(model.importance)


tr = model.simulate(key, args)


from typing import Any, TypeAlias
from genjax import ChoiceMapBuilder as C
SparseGPSModelTrace: TypeAlias = Any




def get_cameras(tr: SparseGPSModelTrace):
    # TODO: Should we leave it like that or grab it from the choice addresses
    latent, obs = tr.get_retval()
    return latent["camera_pose"]


def set_camera_choice(t, cam: Pose, ch=None):
    if ch is None: ch = C.n()
    if t == Ellipsis:
        ch = ch.merge(C["particle_dynamics", "state0", "initial_camera_pose"].set(cam[0]))
        ch = ch.merge(C["particle_dynamics", "states1+", jnp.arange(cam.shape[0]-1), "camera_pose"].set(cam[1:]))
    else:
        if t == 0:
            ch = ch.merge(C["particle_dynamics", "state0", "initial_camera_pose"].set(cam))
        elif t > 0:
            ch = ch.merge(C["particle_dynamics", "states1+", t-1, "camera_pose"].set(cam))
    return ch


def set_sensor_coordinates_choice(t, uvs, ch=None):
    if ch is None: ch = C.n()
    addrs_0 = ["initial_observation", "sensor_coordinates"]
    addrs_T = ["observation", "sensor_coordinates"]
    if t == Ellipsis:
        ch = ch.merge(C["particle_dynamics", "state0", "initial_observation", "sensor_coordinates"].set(uvs[0])
        ch = ch.merge(C["particle_dynamics", "states1+", jnp.arange(uvs.shape[0]-1), "initial_observation", "sensor_coordinates"].set(cam[1:]))
    else:
        if t == 0:
            ch = ch.merge(C["particle_dynamics", "state0", "initial_camera_pose"].set(cam))
        elif t > 0:
            ch = ch.merge(C["particle_dynamics", "states1+", t-1, "camera_pose"].set(cam))
    return ch