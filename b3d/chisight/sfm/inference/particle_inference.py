import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import logpdf as normal_logpdf
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
import genjax

# # # # # # # # # # # # # # # # # # # # # # # # 
# 
#   Utils
# 
# # # # # # # # # # # # # # # # # # # # # # # # 
def rotation_from_first_column(key, a):
    b = jnp.cross(a, jax.random.normal(key, (3,)))
    c = jnp.cross(a, b)

    a = a/jnp.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    b = b/jnp.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    c = c/jnp.sqrt(c[0]**2 + c[1]**2 + c[2]**2)

    return jnp.stack([a,b,c], axis=1)


# # # # # # # # # # # # # # # # # # # # # # # # 
# 
#   Proposals and targets
# 
# # # # # # # # # # # # # # # # # # # # # # # # 
class CylinderParticleProposal():
    def weighted_sample(self, key, y, cam, intr, sig):

        # Compute the ray through y
        x_ = camera_from_screen_and_depth(y, jnp.array(10.), intr)
        B = rotation_from_first_column(key, x_)

        key = jax.random.split(key)[1]
        s = jax.random.uniform(key, (3,), minval=jnp.zeros(3), maxval=jnp.ones(3))

        B = jnp.array([[intr.far, sig, sig]])*B

        w = jnp.array(0.0)
        return w, cam(B@s)
    
cylinder_particle_proposal = CylinderParticleProposal()

class SingleParticleProposal():
    def weighted_sample(self, key, y, cam, intr, sig):

        # Add sensor noise
        key = jax.random.split(key)[1]
        eps = sig*jax.random.normal(key,(2,))
        y = y + eps

        # Unproject with random depth along rays through y
        key = jax.random.split(key)[1]
        z = jax.random.uniform(key, minval=intr.near, maxval=intr.far)
        x = cam(camera_from_screen_and_depth(y, z, intr))

        # Compute the log scores
        w  = - jnp.log(intr.far - intr.near)
        w += normal_logpdf(eps, jnp.zeros(2), jnp.array([sig, sig])).sum()

        return w, x

def particle_vector_proposal_ti(key, t, i, ys, cams, intr, sig):
        
        T, N = ys.shape[:2]
        y   = ys[t,i]
        cam = cams[t]

        # Add sensor noise
        key = jax.random.split(key)[1]
        eps = sig*jax.random.normal(key,(2,))
        y = y + eps

        # Unproject with random depth along rays through y
        key = jax.random.split(key)[1]
        z = jax.random.uniform(key, minval=intr.near, maxval=intr.far)
        x = cam(camera_from_screen_and_depth(y, z, intr))

        # Compute the log scores
        w  = - jnp.log(intr.far - intr.near)
        w += normal_logpdf(eps, jnp.zeros(2), jnp.array([sig, sig])).sum()

        return w, x

def particle_vector_proposal_i(key, i, vis, ys, cams, intr, sig):
    t = jax.random.categorical(key, jnp.log(jnp.where(vis[:,i], 1, 0)))
    return particle_vector_proposal_ti(key, t, i, ys, cams, intr, sig)


def particle_vector_proposal_t(key, t, ys, cams, intr, sig):
        T, N = ys.shape[:2]
        ys_t  = ys[t]
        cam_t = cams[t]

        # Add sensor noise
        key = jax.random.split(key)[1]
        eps = sig*jax.random.normal(key,(N,2))
        ys_t = ys_t + eps

        # Unproject with random depth along rays through y
        key = jax.random.split(key)[1]
        zs = jax.random.uniform(key, (N,), minval=intr.near, maxval=intr.far)
        xs = cam_t(camera_from_screen_and_depth(ys_t, zs, intr))

        # Compute the log scores
        ws  = jnp.repeat(- jnp.log(intr.far - intr.near), N)
        ws += normal_logpdf(eps, jnp.zeros((N,2)), jnp.tile(sig, (N,2))).sum(1)

        return ws, xs

# TODO: The score is approximate only, because we're actually sampling from a mixture over t.
#   We have to add the weights for all the other t's.
def particle_vector_proposal(key, vis, ys, cams, intr, sig):
    """
    Get an approximate weighted posterior sample for each keypoint.

    To be more preceise, for each keypoint `i` sample `t` where `i` is visible and get a sample
    from the approximate posterior `P(x_i | y_ti, c_t )`.

    Args:
        key: Random key
        vis: Visibility mask (T,N)
        ys: Observed 2D keypoints (T,N,2)
        cams: Camera poses (T,)
        intr: Camera intrinsics
        sig: Sensor noise

    Returns:
        ws: Importance weights for each keypoint (N,)
        xs: Particle positions (N,3)
    """
    N = ys.shape[1]
    _,key = jax.random.split(key)
    keys = jax.random.split(key, N)
    return jax.vmap(particle_vector_proposal_i, (0, 0,None,None,None,None,None))(
        keys, jnp.arange(N), vis, ys, cams, intr, sig)




# # # # # # # # # # # # # # # # # # # # # # # # 
# 
#   Geometry
# 
# # # # # # # # # # # # # # # # # # # # # # # # 

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


