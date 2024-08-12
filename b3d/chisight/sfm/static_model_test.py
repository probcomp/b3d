import jax
import jax.numpy as jnp
from b3d.camera import Intrinsics
from b3d.pose import Pose
import genjax
from genjax import Pytree
from genjax import ChoiceMapBuilder as C
from b3d.chisight.sfm.static_model import (
    observation_model, 
    static_model,
    get_observation_scores,
    get_observation_values,    
)


key = jax.random.PRNGKey(0)


# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   Observation Model
# 
# # # # # # # # # # # # # # # # # # # # # # # #
T = 3
N = 5
xs = jnp.array([
    [0.,0.,0.],
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [0.5,0.5,1.],
]) 
vis = jnp.ones((3,5))

cams = Pose.stack_poses([
    Pose.from_position_and_target(jnp.array([ 0.5, -2., 0.0]), jnp.zeros(3)),
    Pose.from_position_and_target(jnp.array([-1.0, -2., 0.0]), jnp.zeros(3)),
    Pose.from_position_and_target(jnp.array([ 2.0, -2., 0.0]), jnp.zeros(3)),
])

intr= Intrinsics(100,100,50.,50.,50.,50.,1e-2,1e2)

obs_model_args =(
    vis,
    xs,
    cams,
    intr,
    jnp.array(2.0), # Observation Noise (pixel scale)
)

key = jax.random.split(key)[1]
tr = observation_model.simulate(key, obs_model_args)
ys = tr.get_retval()

print(f"""
Observation model y scores: 
{get_observation_scores(tr)}
""")

# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   Full Model
# 
# # # # # # # # # # # # # # # # # # # # # # # #
model_args = (
    Pytree.const(T),
    Pytree.const(N),
    Pose.id(),          # Camera init
    intr,               # Camera intrinsice
    jnp.array(1.0),     # Camera positional radius in R3
    jnp.array(0.5),     # Camera rotational radius in SO(3)
    -20.*jnp.ones(3),   # Particle min coords
    20.*jnp.ones(3),    # Particle min coords
    jnp.array(2.0),     # Observation Noiise
)


ch = C["particle_positions", jnp.arange(N)].set(xs)
ch = ch.at["observations", jnp.arange(T), jnp.tile(jnp.arange(N)[None], (T,1)), "sensor_coordinates"].set(ys)
ch = ch.at["visibility", jnp.arange(T), jnp.tile(jnp.arange(N)[None], (T,1))].set(jnp.where(vis,1,0))
ch = ch.at["camera_poses", jnp.arange(T), "pose"].set(cams)

key = jax.random.split(key)[1]
tr,_ = static_model.importance(key, ch, model_args)

print(f"""
Full (static) model y scores: 
{get_observation_scores(tr)}
""")