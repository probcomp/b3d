import jax
import jax.numpy as jnp
import genjax
from genjax import Pytree
from genjax import ChoiceMapBuilder as C
from genjax import Target, smc
from b3d.pose import Pose, uniform_pose_in_ball
from b3d.types import Array, Float
from b3d.camera import (
    Intrinsics,
    screen_from_world,
    camera_from_screen_and_depth,
)
from b3d.utils import keysplit


# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   The Static Model
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
def camera_prior(T:Pytree.const, c0:Pose, rx, rq):
    T = T.const
    _, cams = genjax.scan(n=T)(camera_motion_model)((c0, rx, rq), None) @ "camera_poses"
    return cams


@genjax.gen
def particle_prior(N:Pytree.const, particle_min, particle_max):
    """Uniform prior over particle positions."""
    N = N.const
    xs = genjax.repeat(n=N)(genjax.uniform)(particle_min, particle_max) @ "particle_positions"
    return xs


@genjax.gen
def visibility_model(T:Pytree.const, N:Pytree.const):
    T = T.const
    N = N.const
    # NOTE: Bernoulli takes *actual* logits, i.e. log(p/(1-p))
    vis = genjax.bernoulli.vmap(in_axes=(0,)).vmap(in_axes=(0,))(jnp.tile(jnp.log(0.5),(T,N))) @ "visibility"
    return vis


@genjax.gen
def single_observation_model(x:Array, cam:Pose, intr, sig):
    y_ = screen_from_world(x, cam, intr, culling=True)
    y  = genjax.normal(y_, jnp.array([sig,sig])) @ "sensor_coordinates"
    return y


masked_single_observation_model = single_observation_model.mask()


@genjax.gen
def observation_model(vis:Array, xs:Array, cams:Pose, intr:Intrinsics, sig:Float):
    """
    Example:
    ```
    ...
    ```
    """
    masked_ys = masked_single_observation_model.vmap(in_axes=(0,0,None,None,None)
                ).vmap(in_axes=(0,None,0,None,None)
                    )(vis == 1, xs, cams, intr, sig) @ "observations"
    return masked_ys.value


@genjax.gen
def static_model(
        T:Pytree.const, 
        N:Pytree.const, 
        c0:Pose,
        intr:Intrinsics, 
        cam_rx:Float, 
        cam_rq:Float,
        particle_min:Array,
        particle_max:Array,
        obs_sig:Float, 
    ):
    """
    Choices:
        `particle_positions`: FloatArray (N,3)
        `camera_poses`: Pose (T,)
        `visibility`: FloatArray(T,N)
        `observations`: FloatArray(T,N,2)
    """
    # Inlined address @ "particle_positions" 
    xs = particle_prior.inline(N, particle_min, particle_max) 

    # Inlined address @ "camera_poses"
    cams = camera_prior.inline(T, c0, cam_rx, cam_rq) 

    # Inlined address @ "visibility"
    vis = visibility_model.inline(T,N)

    # Inlined address @ "observations"
    ys = observation_model.inline(vis == 1, xs, cams, intr, obs_sig)

    return {
        "particle_positions": xs, 
        "camera_poses": cams,
        "observations": ys,
        "visibility": vis,
    }


# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   Test Model
# 
# # # # # # # # # # # # # # # # # # # # # # # #
def get_a_test_model(T,N):
    pass


# # # # # # # # # # # # # # # # # # # # # # # #
# 
#   Helper
# 
# # # # # # # # # # # # # # # # # # # # # # # #
def get_particle_position_values(tr): return tr.get_choices()("particle_positions").c.v
def get_camera_pose_values(tr): return tr.get_choices()("camera_poses").c("pose").v
def get_visibility_values(tr): return tr.get_choices()("visibility").c.c.v
def get_visibility_values_bool(tr): return get_visibility_values(tr) == 1
def get_observation_values(tr): return tr.get_choices()("observations").c.c("sensor_coordinates").c.v

# NOTE: Workarounds till `tr.project` works
def get_particle_position_scores(tr): 
    i = tr.addresses.visited.index(("particle_positions",))
    return tr.subtraces[i].inner.inner.inner.score

def get_camera_pose_scores(tr): 
    i = tr.addresses.visited.index(("camera_poses",))
    return tr.subtraces[i].inner.subtraces[0].score

def get_visibility_scores(tr): 
    i = tr.addresses.visited.index(("visibility",))
    return tr.subtraces[i].inner.inner.score

def get_observation_scores(tr): 
    i = tr.addresses.visited.index(("observations",))
    return tr.subtraces[i].inner.inner.inner.score

def get_masked_observation_scores(tr): 
    vis = get_visibility_values_bool(tr)
    return jnp.where(vis, get_observation_scores(tr), 0.0)

def get_individual_particle_posterior_target_scores(tr):
    """
    $$
        \log p(x_*) + \sum_t \log P( y_{t*} \mid x_*, c_t, v_t*) + \sum_t \log P(v_{t*})
    $$
    """
    return (
        get_particle_position_scores(tr) +
        get_visibility_scores(tr).sum(0) + 
        get_masked_observation_scores(tr).sum(0)
    )