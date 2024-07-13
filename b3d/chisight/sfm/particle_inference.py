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


# TODO: Check this. ChatGPT spit that out.
def closest_points_on_lines(x, v, x_prime, v_prime):
    """
    Given two affine lines computes point on each line 
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
        
    t = (a_dot_b * b_dot_w0 - b_dot_b * a_dot_w0) / denom
    s = (a_dot_a * b_dot_w0 - a_dot_b * a_dot_w0) / denom
    
    # Calculate the closest points on the lines
    p1 = x + t * a
    p2 = x_prime + s * b
    
    return (p1, p2)


def _latent_keypoint_from_lines(u0, u1, cam0, cam1, intr):
    """
    Returns keypoint that is closest to both keypoint lines.
    """
    x0 = world_from_screen(u0, cam0, intr)
    x1 = world_from_screen(u1, cam1, intr)

    a, b = closest_points_on_lines(
        cam0.pos, x0 - cam0.pos, 
        cam1.pos, x1 - cam1.pos)
    
    x = (a + b)/2

    return x



# # # # # # # # # # # # # # # # # # # # # # # # 
# 
#   Gaussian Inference
# 
# # # # # # # # # # # # # # # # # # # # # # # # 
from b3d.chisight.gps_utils import (
    gaussian_pdf_product, 
    gaussian_pdf_product_multiple,
    cov_from_dq_composition
)


def rotation_from_first_column(key, a):
    b = jnp.cros(a, jax.random.normal(key, (3,)))
    c = jnp.cross(a, b)

    a = a/jnp.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    b = b/jnp.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    c = b/jnp.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

    return jnp.stack([a,b,c], axis=1)


def gaussian_from_keypoint(z, diag, u, cam, intr):
    x = camera_from_screen_and_depth(u, z, intr)
    p = Pose.from_position_and_target(jnp.zeros(3), x, up=jnp.array([1.,0.,0.]))
    quat = (cam@p).quat
    mu = cam(x)
    cov = cov_from_dq_composition(diag, quat)
    return mu, cov


def _gaussian_keypoint_posterior(z0, z1, diag0, diag1, u0, u1, cam0, cam1, intr):
    mu0, cov0 = gaussian_from_keypoint(z0, diag0, u0, cam0, intr)
    mu1, cov1 = gaussian_from_keypoint(z1, diag1, u1, cam1, intr)
    mu, cov = gaussian_pdf_product(mu0, cov0, mu1, cov1)
    return mu, cov


