import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import logpdf as normal_logpdf
from b3d.camera import Intrinsics, screen_from_world
from b3d.pose import Pose, uniform_pose_in_ball
from jax.scipy.special import logsumexp
from typing import TypeAlias


Array: TypeAlias = jax.Array
Float: TypeAlias = Array
Int: TypeAlias = Array
ImportanceSample: TypeAlias = Array
ImportanceWeight: TypeAlias = Array
SensorCoordinates: TypeAlias = Array
LatentKeypoint: TypeAlias = Array


def uniform_motion_step_score(p:Pose, p0:Pose, rx:Float, rq:Float):
    """log P(c | c')"""
    return uniform_pose_in_ball.logpdf(p, p0, rx, rq)


def single_observation_score(y:SensorCoordinates, x:LatentKeypoint, cam:Pose, intr, sig, culling=True):
    """log P(y | x, c)"""
    y_ = screen_from_world(x, cam, intr, culling=culling)
    w  = normal_logpdf(y, y_, jnp.array([sig,sig])).sum()
    return w


def maker_observation_scorer(single_score):

    def observation_score(ys:SensorCoordinates, xs:LatentKeypoint, cams:Pose, intr, *args):
        """P(ys | xs, cs)"""
        observation_scores_over_i  = jax.vmap(single_score, (0, 0, None, None) + (None,)*len(args))
        observation_scores_over_ti = jax.vmap(observation_scores_over_i, (0, None, 0, None) + (None,)*len(args))
        return observation_scores_over_ti(ys, xs, cams, intr, *args)
    
    return observation_score


def make_motion_scorer(step_logpdf):

    def motion_score(ps, p0, *args):
        """log P(cs)"""
        w0 = step_logpdf(ps[0], p0, *args)
        ws = jax.vmap(step_logpdf, (0,0)+(None,)*len(args))(ps[1:], ps[:-1], *args)
        return jnp.concatenate([w0[None], ws])
    
    return motion_score


uniform_motion_scores = make_motion_scorer(uniform_motion_step_score)
observation_scores = maker_observation_scorer(single_observation_score)


def uniform_particle_scores(xs, minvals, maxvals):
    """log P(xs)"""
    return jnp.repeat(-jnp.log(maxvals - minvals).sum(), xs.shape[0])

def make_camera_posterior_scorer(
        motion_scores, 
        motion_args, 
        observation_scores, 
        observation_args
    ):
    
    def camera_posterior_score(
            cs:Pose, 
            ys:SensorCoordinates, 
            ws:ImportanceWeight, 
            xs:ImportanceSample):
        """
        Approximation of $\log P(c \mid y)$.

        Given a collection of weighted particle (posterior) samples $\big\{ (w_{ij}, x_{ij}) \big\}_{I\times J}$, 
        we can compute an approximate log score as follows:
        $$
        \log P(c \mid y) \stackrel{\propto}{\approx} \log P(c) - N\log S + \sum_i \ \log \sum_j \exp \Big( \log w_{ij} + \sum_t \log \ell_{t,ij}  \Big).
        $$
        """
        T, N = ys.shape[:2]
        S = xs.shape[0]
        w = (
            motion_scores(cs, *motion_args) - 
            N*jnp.log(S) + 
            jnp.sum(logsumexp(jax.vmap(observation_scores, (None,0,None)+(None,)*len(observation_args))(
                ys, xs, cs, *observation_args).sum(1), axis=0))
        )
        return w
    
    return camera_posterior_score
