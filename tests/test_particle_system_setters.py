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
