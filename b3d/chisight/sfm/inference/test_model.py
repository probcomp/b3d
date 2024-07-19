import jax
import jax.numpy as jnp
import genjax
from genjax import Pytree
from genjax import ChoiceMapBuilder as C
from genjax import Target, smc
from b3d.pose import Pose, uniform_pose_in_ball
from b3d.camera import (
    screen_from_world,
    camera_from_screen_and_depth,
)
from b3d.utils import keysplit



# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   The Model
# 
# # # # # # # # # # # # # # # # # # # # # # # #
# TODO: Can we make it easy to fix args in a model?
@genjax.gen
def camera_motion_model(carry, _):
    p, rx, rq = carry
    p = uniform_pose_in_ball(p, rx, rq) @ "pose"
    carry = (p, rx, rq)
    return carry, p

@genjax.gen
def single_observation_model(x, cam, intr, sig):
    y_ = screen_from_world(x, cam, intr, culling=True)
    y  = genjax.normal(y_, jnp.array([sig,sig])) @ "sensor_coordinates"
    return y

masked_single_observation_model = single_observation_model.mask()

@genjax.gen
def model(
        T, 
        N, 
        intr, 
        c0 , 
        obs_sig, 
        cam_rx, 
        cam_rq,
        particle_min,
        particle_max,
    ):
    """"
    Choices:
        `particle_positions`: FloatArray (N,3)
        `camera_poses`: Pose (T,)
        `visibility`: FloatArray(T,N)
        `observations`: FloatArray(T,N,2)
    """
    
    T = T.const
    N = N.const

    xs = genjax.repeat(n=N)(genjax.uniform)(particle_min, particle_max) @ "particle_positions"
    _, cams = genjax.scan(n=T)(camera_motion_model)((c0, cam_rx, cam_rq), None) @ "camera_poses"

    # NOTE: Bernoulli takes *actual* logits, i.e. log(p/(1-p))
    vis = genjax.bernoulli.vmap(in_axes=(0,)).vmap(in_axes=(0,))(jnp.tile(jnp.log(0.5),(T,N))) @ "visibility"

    masked_ys = masked_single_observation_model.vmap(in_axes=(0,0,None,None,None)
                ).vmap(in_axes=(0,None,0,None,None)
                    )(vis == 1, xs, cams, intr, obs_sig) @ "observations"
    ys = masked_ys.value

    
    return {
        "particle_positions": xs, 
        "camera_poses": cams,
        "observations": ys,
        "visibility": vis,
    }

# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   Helper
# 
# # # # # # # # # # # # # # # # # # # # # # # #
def get_particle_values(tr): return tr.get_choices()("particle_positions").c.v
def get_camera_values(tr): return tr.get_choices()("camera_poses").c("pose").v
def get_visibility_values(tr): return tr.get_choices()("visibility").c.c.v
def get_visibility_values_bool(tr): return get_visibility_values(tr) == 1
def get_observation_values(tr): return tr.get_choices()("observations").c.c("sensor_coordinates").c.v


# NOTE: Workarounds till `tr.project` works
def get_particle_scores(tr): return tr.subtraces[0].inner.inner.inner.score
def get_camera_scores(tr): return tr.subtraces[1].inner.subtraces[0].score
def get_visibility_scores(tr): return tr.subtraces[2].inner.inner.score
def get_observation_scores(tr): return tr.subtraces[3].inner.inner.inner.score
def get_individual_particle_posterior_target_scores(tr):
    """
    $$
        \sum_t P( y_{t*} \mid x_*, c_t)
    $$
    """
    vis = get_visibility_values_bool(tr)
    return (
        get_particle_scores(tr) +
        get_visibility_scores(tr).sum(0) + 
        jnp.where(vis, get_observation_scores(tr), 0.0).sum(0)
    )


