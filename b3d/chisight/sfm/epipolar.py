import jax
import jax.numpy as jnp
from b3d.pose import Pose, Rot
from b3d.camera import (
    screen_from_world,
    screen_from_screen,
    world_from_screen,
)
from b3d.utils import keysplit, Bunch

# # # # # # # # # # # # # # # # # # # # 
# 
#   Epipolar geometry
# 
# # # # # # # # # # # # # # # # # # # # 

def get_epipoles(cam0, cam1, intr):
    """Get epipoles of two cameras."""
    e0 = screen_from_world(cam1.pos, cam0, intr)
    e1 = screen_from_world(cam0.pos, cam1, intr)
    return jnp.stack([e0, e1], axis=0)

def get_epipole(cam, intr):
    """Get epipole of a camera with respect to fixed standard camera (at origin)."""
    e = screen_from_world(jnp.zeros(3), cam, intr)
    return e

def dist_to_line(u, l):
    """Computes the distance of a 2D-point to a 2D Line through the origin."""
    l = l/jnp.sqrt(l[...,[0]]**2 + l[...,[1]]**2)
    il = jnp.stack([-l[...,1],l[...,0]], axis=-1)
    d = u[...,0]*il[...,0] + u[...,1]*il[...,1] 
    d = jnp.abs(d)
    return d

# TODO: This is experimental.
def dist_to_ray(u, l, outlier_val = 1e6):
    l = l/jnp.sqrt(l[...,[0]]**2 + l[...,[1]]**2)
    il = jnp.stack([-l[...,1],l[...,0]], axis=-1)
    d = u[...,0]*il[...,0] + u[...,1]*il[...,1] 
    front = u[...,0]*l[...,0] + u[...,1]*l[...,1] 
    d = jnp.where(front > 1e2, jnp.abs(d), outlier_val)
    return d

def _epi_scorer(cam, u0, u1, intr):
    """
    Computes the distances of `u1` to the epipolar lines induced by `u0` and `cam`.
    """
    e = get_epipole(cam, intr)

    l = screen_from_screen(u0, cam, Pose.id(), intr)
    l = l - e
    u = u1 - e

    # TODO: Constrain so that we only consider the 
    #   positive part of the line. That "should" (might) get rid of 
    #   weird local maxima with points behind the camera. 
    d = dist_to_ray(u, l)

    return d


@jax.jit
def epi_scorer(cams, uv0, uv1, intr):
    """
    Computes epipolar scores for an array of cameras with respect to two
    aligned 2D-keypoint arrays.
    """
    return jax.vmap(
        lambda cam, uv0, uv1, intr: _epi_scorer(cam, uv0, uv1, intr), 
        (0,None,None,None)
    )(cams, uv0, uv1, intr)  

# # # # # # # # # # # # # # # # # # # # 
# 
#   Helper
# 
# # # # # # # # # # # # # # # # # # # # 

# TODO: Check this. ChatGPT spit that out.
def closest_points_on_lines(x, v, x_prime, v_prime):
    """
    Given two affine lines computes a point on each line 
    with minimal distance between them.
    """
    # Define the direction vectors
    a = v
    b = v_prime
    
    # Define the vector between the two points on the lines
    w0 = x - x_prime
    
    # Calculate coefficients for the system of linear equations
    a_dot_a = jnp.dot(a, a)
    b_dot_b = jnp.dot(b, b)
    a_dot_b = jnp.dot(a, b)
    a_dot_w0 = jnp.dot(a, w0)
    b_dot_w0 = jnp.dot(b, w0)
    
    # Solving the system of linear equations for t and s
    denom = a_dot_a * b_dot_b - a_dot_b * a_dot_b
    
    # If the denominator is zero, the lines are parallel
    # if jnp.isclose(denom, 0):
        # raise ValueError("The lines are parallel and do not have a unique minimal distance.")
    
    t = (a_dot_b * b_dot_w0 - b_dot_b * a_dot_w0) / denom
    s = (a_dot_a * b_dot_w0 - a_dot_b * a_dot_w0) / denom
    
    # Calculate the closest points on the lines
    p1 = x + t * a
    p2 = x_prime + s * b
    
    return p1, p2

# TODO/NOTE: closest point is not always the best, one should grab the projection
#   onto the epilines -- at least try that.
def latent_keypoints_from_cameras(uvs0, uvs1, cam0, cam1, intr):
    """
    Naive inference of 3d keypoints from two cameras and sensor keypoints.
    """

    xs0 = world_from_screen(uvs0, cam0, intr)
    xs1 = world_from_screen(uvs1, cam1, intr)

    a, b = jax.vmap(closest_points_on_lines, (None,0,None,0))(
        cam0.pos, xs0 - cam0.pos[None], 
        cam1.pos, xs1 - cam1.pos[None])
    
    xs = (a + b)/2

    return xs

