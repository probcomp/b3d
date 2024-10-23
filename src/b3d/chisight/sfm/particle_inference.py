import jax

from b3d.camera import camera_from_screen_and_depth, screen_from_camera
from b3d.pose import Pose
from b3d.utils import keysplit


def _single_view_score(x, p: Pose, u, sigma, intr):
    """
    Returns the log-likelihood of a world point
    given a camera pose and image coordinates.

    Args:
        x: World point
        p: Camera pose
        u: Image coordinates
        sigma: Image noise
        intr: Camera intrinsics

    Returns:
        Log-likelihood of the world point
    """
    # Compute the depth log-score
    zmin = intr.near
    zmax = intr.far
    logp_z = -(zmax - zmin)

    # Project the worldpoint into the image
    # and compute the sensor log-score
    u_ = screen_from_camera(p.inv()(x), intr)
    logp_u = jax.scipy.stats.norm.logpdf(u_, loc=u, scale=sigma).sum()

    # Combine the scores
    logp_x = logp_u + logp_z
    return logp_x


def single_view_score(xs, p: Pose, uvs, sigma, intr):
    """
    Returns the log-likelihood of an array of world points
    given a camera pose and their corresponding image coordinates.

    Args:
        xs: Array of world points
        p: Camera pose
        uvs: Array of image coordinates
        sigma: Image noise
        intr: Camera intrinsics

    Returns:
        Array of log-likelihoods
    """
    return jax.vmap(_single_view_score, (0, None, 0, None, None))(
        xs, p, uvs, sigma, intr
    )


def _single_view_sample(key, uv, p, sigma, intr):
    """
    Returns a world point sample given image coordinates from a single view.

    Args:
        key: JAX random key
        uv: 2D image coordinates
        p: Camera pose
        sigma: Image noise
        intr: Camera intrinsics

    Returns:
        World point and its log-likelihood
    """
    zmin = intr.near
    zmax = intr.far

    # Sample noise in image coordinates
    eps = sigma * jax.random.normal(key, (2,))
    logp_eps = jax.scipy.stats.norm.logpdf(eps, loc=0, scale=sigma).sum()

    # Sample depth
    z = jax.random.uniform(key, minval=zmin, maxval=zmax)
    logp_z = -(zmax - zmin)

    # Compute the world point, and its log score
    x = p(camera_from_screen_and_depth(uv + eps, z, intr))
    logp_x = logp_eps + logp_z
    return x, logp_x


def single_view_sample(key, uvs, p, sigma, intr):
    """
    Returns a world point sample for each given image coordinate from a single view.

    Args:
        key: JAX random key
        uvs: Array of 2D image coordinates
        p: Camera pose
        sigma: Image noise
        intr: Camera intrinsics

    Returns:
        Array of world points and their log-likelihoods
    """
    _, keys = keysplit(key, 1, len(uvs))
    return jax.vmap(_single_view_sample, (0, 0, None, None, None))(
        keys, uvs, p, sigma, intr
    )
