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
from jax.scipy.linalg import inv


# TODO: Check that
def gaussian_pdf_product(mean1, cov1, mean2, cov2):
    """
    Computes the product of two 3D Gaussian PDFs.

    Args:
        mean1: Mean vector of the first Gaussian (3-dimensional).
        cov1: Covariance matrix of the first Gaussian (3x3 matrix).
        mean2: Mean vector of the second Gaussian (3-dimensional).
        cov2: Covariance matrix of the second Gaussian (3x3 matrix).
    
    Returns:
        mean_prod: Mean vector of the product Gaussian.
        cov_prod: Covariance matrix of the product Gaussian.

    
    """    
    # Someone bless the internet:
    # > https://math.stackexchange.com/questions/157172/product-of-two-multivariate-gaussians-distributions
    cov_prod  = inv(inv(cov1) + inv(cov2)) 
    mean_prod = cov_prod @ (inv(cov1) @ mean1 + inv(cov2) @ mean2)
    
    return mean_prod, cov_prod


# TODO: Check that
def gaussian_pdf_product_multiple(means, covariances):
    """
    Computes the product of multiple 3D Gaussian PDFs.
    
    Args:
    means: A 2D array where each row is a mean vector of a Gaussian (N x 3).
    covariances: A 3D array where each slice along the first dimension is a 3x3 covariance matrix (N x 3 x 3).
    
    Returns:
    mean_prod: Mean vector of the product Gaussian.
    cov_prod: Covariance matrix of the product Gaussian.
    """
    # Convert means and covariances to JAX arrays
    means = jnp.asarray(means)
    covariances = jnp.asarray(covariances)
    
    # Compute the inverse of each covariance matrix
    inv_covariances = jax.vmap(inv)(covariances)
    
    # Sum of the inverses of covariance matrices
    cov_prod_inv = jnp.sum(inv_covariances, axis=0)
    
    # Compute the product covariance matrix
    cov_prod = inv(cov_prod_inv)
    
    # Compute the weighted sum of means
    weighted_means_sum = jnp.sum(jax.vmap(lambda inv_cov, mean: inv_cov @ mean)(inv_covariances, means), axis=0)
    
    # Compute the product mean vector
    mean_prod = cov_prod @ weighted_means_sum
    
    return mean_prod, cov_prod


def rotation_from_first_column(key, a):
    b = jnp.cros(a, jax.random.normal(key, (3,)))
    c = jnp.cross(a, b)

    a = a/jnp.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    b = b/jnp.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    c = b/jnp.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

    return jnp.stack([a,b,c], axis=1)


from b3d.pose import Pose, Rot
def cov_from_dq_composition(diag, quat):
    """
    Covariance matrix from particle representation `(diag, quat)`,
    where `diag` is an array of eigenvalues and `quat` is a quaternion
    representing the matrix of eigenvectors.
    """
    U = Rot.from_quat(quat).as_matrix()
    C = U @ jnp.diag(diag) @ U.T
    return C


def gaussian_from_keypoint(z, diag, u, cam, intr):
    x = cam(camera_from_screen_and_depth(u, z, cam, intr))
    p = Pose.from_position_and_target(cam.pos, x)
    cov = cov_from_dq_composition(diag, p.quat)
    return x, cov


def _gaussian_keypoint_posterior(z0, z1, diag0, diag1, u0, u1, cam0, cam1, intr):
    mu0, cov0 = gaussian_from_keypoint(z0, diag0, u0, cam0, intr)
    mu1, cov1 = gaussian_from_keypoint(z1, diag1, u1, cam1, intr)
    mu, cov = gaussian_pdf_product(mu0, cov0, mu1, cov1)
    return mu, cov