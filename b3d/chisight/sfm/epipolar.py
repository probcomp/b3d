import jax
import jax.numpy as jnp
from b3d.pose import Pose, Rot
from b3d.camera import (
    screen_from_world,
    screen_from_camera,
    camera_from_screen,
    world_from_screen,
    camera_from_screen_and_depth,
)
from b3d.utils import keysplit
from sklearn.utils import Bunch


# # # # # # # # # # # # # # # # # # # # 
# 
#   Epipolar geometry
# 
# # # # # # # # # # # # # # # # # # # # 

def get_epipole(cam, intr):
    """Get epipole of a camera with respect to fixed standard camera (at origin)."""
    e = screen_from_world(jnp.zeros(3), cam, intr)
    return e

def get_epipoles(cam0, cam1, intr):
    """Get epipoles of two cameras."""
    e0 = screen_from_world(cam1.pos, cam0, intr)
    e1 = screen_from_world(cam0.pos, cam1, intr)
    return jnp.stack([e0, e1], axis=0)

def dist_to_line(u, l):
    """
    Returns the distance of 'u' to the line through 'l'.
    """
    # Normalize and 
    # rotate by 90 degrees
    l = l/jnp.sqrt(l[...,[0]]**2 + l[...,[1]]**2)
    il = jnp.stack([-l[...,1],l[...,0]], axis=-1)
    d = u[...,0]*il[...,0] + u[...,1]*il[...,1] 
    return jnp.abs(d)


def dist_to_and_along_line(u, l):
    """
    Returns the distance of 'u' to the line through 'l', and 
    the amount that of `u` along `l/|l|`, that is,

        `|dot(u, il/|il|)|` and `dot(u, l/|l|)`.
    """
    # Normalize and 
    # rotate by 90 degrees
    l = l/jnp.sqrt(l[...,[0]]**2 + l[...,[1]]**2)
    il = jnp.stack([-l[...,1],l[...,0]], axis=-1)
    d = u[...,0]*il[...,0] + u[...,1]*il[...,1] 
    s = u[...,0]* l[...,0] + u[...,1]* l[...,1] 
    return jnp.abs(d), s


def _epi_constraint(cam, u0, u1, intr):
    """
    Textbook epipolar constraint. Computes the (unsigned) alignment between the epipolar planes 
    spanned by `u0`, `u1`, and `cam`; zero means perfectly aligned. 

    Args:
        cam: Relative camera Pose
        u0: Array of shape (..., 2)
        u1: Array of shape (..., 2) (same shape as `u0`)
        intr: Intrinsics
    
    Returns
        Array of shape (...)
    """
    # TODO: Add a reference.
    # NOTE: We work with a relative pose here, that is, we assume
    #       at time 0 world and camera frames are the same.
    v0 = camera_from_screen(u0, intr)
    v1 = world_from_screen(u1, cam, intr) - cam.pos
    c = cam.pos

    # Normalize
    v0 = v0/jnp.sqrt(v0[...,[0]]**2 + v0[...,[1]]**2 + v0[...,[2]]**2)
    v1 = v1/jnp.sqrt(v1[...,[0]]**2 + v1[...,[1]]**2 + v1[...,[2]]**2)
    c = c/jnp.sqrt(c[...,[0]]**2 + c[...,[1]]**2+ c[...,[2]]**2)
    n = jnp.cross(v0, c[None], axis=-1)
    n = n/jnp.sqrt(n[...,[0]]**2 + n[...,[1]]**2+ n[...,[2]]**2)
    d = (n * v1).sum(-1)

    # Project to epi plane spanned by v0 and c
    v1_ = v1 - (v1*n).sum(-1)[:,None]*n
    v1_ = v1_/jnp.linalg.norm(v1_, axis=-1, keepdims=True)
    h = jnp.abs(d)

    aux = dict(v0=v0, v1=v1, v1_in_epiplane=v1_)
    return h, aux

vmap_epi_constraint = jax.vmap(
        lambda cam, uv0, uv1, intr: _epi_constraint(cam, uv0, uv1, intr)[0], 
        (0,None,None,None)
)


# NOTE: Experimental, don't rely on this
def _epi_constraint_variation_1(cam, u0, u1, intr):
    h, aux = _epi_constraint(cam, u0, u1, intr)[0]
    v0  = aux["v0"]
    v1_ = aux["v1_on_epiplane"]
    c = cam.pos/jnp.linalg.norm(cam.pos)
    h = h - jnp.sign( (v0 * c).sum(-1) - (v1_ * c).sum(-1) ).sum()
    return h, None


def _epi_distance(cam, u0, u1, intr):
    """
    Projected version of epipolar constraints.

    Computes the distances of `u1` to the epipolar line 
    on the sensor canvas induced by `u0` and `cam`.

    Args:
        cam: Relative camera Pose
        u0: Array of shape (..., 2)
        u1: Array of shape (..., 2) (same shape as `u0`)
        intr: Intrinsics
    
    Returns
        Array of shape (...)
    """
    # NOTE: We work with a relative pose here, that is, we assume
    #       at time 0 world and camera frames are the same.
    # TODO: Constrain so that we only consider the 
    #   positive part of the line. That "should" (might) get rid of 
    #   weird local maxima with points behind the camera. 
    #   One should also look at the far end of the line since 
    #   beyond this one cannot reach.

    # Get epipole in frame 1
    e = screen_from_world(jnp.zeros(3), cam, intr)
    
    # Take a point on the ray shooting through u0, 
    # and project onto opposite screen
    x = camera_from_screen(u0, intr)
    v1 = screen_from_world(x, cam, intr)
    l = v1 - e
    u = u1 - e

    d, _ = dist_to_and_along_line(u, l)
    aux = {"epipole": e, "line_direction": l,}
    return d, aux

vmap_epi_distance = jax.vmap(
        lambda cam, uv0, uv1, intr: _epi_distance(cam, uv0, uv1, intr)[0], 
        (0,None,None,None)
)

# # # # # # # # # # # # # # # # # # # # 
# 
#   Debugging
# 
# # # # # # # # # # # # # # # # # # # # 

def _get_epipolar_debugging_data(cam, u0, u1, intr):

    # Get epipole in frame 1
    e = screen_from_world(jnp.zeros(3), cam, intr)
    
    # Take a point on the ray shooting through u0, 
    # and project onto opposite screen
    x = camera_from_screen(u0, intr)
    v1 = screen_from_world(x, cam, intr)
    l = v1 - e
    u = u1 - e
    d, s = dist_to_and_along_line(u, l)

    l_norm = jnp.sqrt(l[...,[0]]**2 + l[...,[1]]**2)

    proj_vec = s[...,None] * l/l_norm
    error_vec = proj_vec - u


    x_near = camera_from_screen_and_depth(u0, jnp.array([intr.near]), intr)
    x_far  = camera_from_screen_and_depth(u0, jnp.array([intr.far]), intr)
    v_near = screen_from_world(x_near, cam, intr)
    v_far  = screen_from_world(x_far, cam, intr)
    vs = jnp.stack([v_near, v_far], axis=1)


    v0 = camera_from_screen(u0, intr)
    v1 = world_from_screen(u1, cam, intr) - cam.pos
    c = cam.pos
    

    # Normalize
    v0 = v0/jnp.sqrt(v0[...,[0]]**2 + v0[...,[1]]**2 + v0[...,[2]]**2)
    v1 = v1/jnp.sqrt(v1[...,[0]]**2 + v1[...,[1]]**2 + v1[...,[2]]**2)
    c = c/jnp.sqrt(c[...,[0]]**2 + c[...,[1]]**2+ c[...,[2]]**2)
    n = jnp.cross(v0, c[None], axis=-1)
    n = n/jnp.sqrt(n[...,[0]]**2 + n[...,[1]]**2+ n[...,[2]]**2)

    return dict(
        epipole = e,
        line_directions = l,
        epi_distance = d,
        epi_scalar = s,
        projection_vector = proj_vec,
        error_vector = error_vec,
        near_far_screen = vs,
        near_far_world = jnp.stack([x_near, x_far], axis=1),
        v0 = v0,
        v1 = v1,
        c = c,
        n = jnp.cross(v0, c[None], axis=-1)
    )


# # # # # # # # # # # # # # # # # # # # 
# 
#   Helper
# 
# # # # # # # # # # # # # # # # # # # # 

def angle(v,w):
  v = v/jnp.linalg.norm(v, axis=-1, keepdims=True)
  w = w/jnp.linalg.norm(w, axis=-1, keepdims=True)
  return jnp.arccos((v*w).sum(-1))


# # # # # # # # # # # # # # # # # # # # 
# 
#   Proposal Factories
# 
# # # # # # # # # # # # # # # # # # # # 
from b3d.pose import uniform_pose_in_ball
vmap_uniform_pose = jax.jit(jax.vmap(uniform_pose_in_ball.sample, (0,None,None,None)))


def make_two_frame_proposal(loss_func):
    """
        Returns a pose proposal, using the following recipe.
         - Sample *uniformly* around target pose, then
         - compute the lossess, and
         - return the the argmin.
    """

    def proposal(key, p0, p1, uvs0, uvs1, intr, rx=1.5, rq=0.25, S=100):
        """
        Return pose a proposal around target pose `p1` as follows:
         - Sample *uniformly* around target pose `p1`, then
         - compute the lossess, and
         - return the the argmin.
        """
        # Create new key branch
        _, key = keysplit(key, 1, 1)

        # Switch to relative poses.
        q = p0.inv() @ p1

        # Sample and score
        # test poses
        key, keys = keysplit(key, 1, S)
        qs = vmap_uniform_pose(keys, q, rx, rq)
        losses_ = jax.vmap(loss_func, (0,None,None,None))(qs, uvs0, uvs1, intr)[0]
        loss = jnp.nan_to_num(losses_.sum(1), nan=jnp.inf)

        # Pick best test pose
        # TODO: Resample?
        i = jnp.argmin(loss)
        q = qs[i]

        aux = {"proposals": qs, "loss": loss, "winner_index": i, "winner_loss": loss[i]}

        return q, aux

    return proposal


# # # # # # # # # # # # # # # # # # # # 
# 
#   Appendix
# 
# # # # # # # # # # # # # # # # # # # # 
# NOTE/TODO: This doesn't work as well as the other scorer. 
#   I am just keeping this for further analysis.
def _epi_scorer_other_version(cam, u0, u1, intr):
    """
    Computes the distances of `u1` to the epipolar lines induced by `u0` and `cam`.
    """
    e = get_epipole(cam, intr)

    x0 = camera_from_screen_and_depth(u0, intr.far*jnp.ones(u0.shape[:-1]), intr)
    l = screen_from_world(x0, cam, intr)
    l_norm = jnp.sqrt(l[...,0]**2 + l[...,1]**2)

    l = l - e
    u = u1 - e

    # TODO: Constrain so that we only consider the 
    #   positive part of the line. That "should" (might) get rid of 
    #   weird local maxima with points behind the camera. 
    d, s = dist_to_and_along_line(u, l)
    d = jnp.where(s >    0.0, d, 1e2)
    d = jnp.where(s < l_norm, d, 1e2)

    s = jnp.clip(s, 0.0, jnp.inf)
    ys = e + s[:,None]*l/l_norm[:,None]

    return d, ys