import jax
import jax.numpy as jnp

from b3d.camera import camera_from_screen_and_depth, screen_from_camera
from b3d.pose import Pose, uniform_pose_in_ball
from b3d.utils import keysplit

vmap_uniform_pose = jax.jit(
    jax.vmap(uniform_pose_in_ball.sample, (0, None, None, None))
)


def random_choices(key, num_choices, shape):
    """
    Vmapedded version of `jax.random.choice`.

    Returns a `shape[0]` samples of `shape[1:]` random
    indices in the range `0` to `num_choices`.

    Example:
    ```
    # Create 1000 samples of 8 random indices
    inds = random_choices(key, 100, (1_000, 8))
    assert inds.shape == (1000,8)
    ```
    """
    assert len(shape) <= 2
    _, keys = keysplit(key, 1, shape[0])
    return jax.vmap(
        lambda key: jax.random.choice(key, num_choices, shape[1:], replace=False)
    )(keys)


def rescale_pose(p, p_true):
    """Rescales the position vector of a pose to match the scale of another pose."""
    scale = jnp.linalg.norm(p_true.pos) / jnp.linalg.norm(p.pos)
    return Pose(scale * p.pos, p.quat)


rescale_poses = jax.vmap(rescale_pose, in_axes=(0, None))


def val_from_im(uv, im):
    """Get the pixel value from an image."""
    return im[uv[1].astype(jnp.int32), uv[0].astype(jnp.int32)]


vals_from_im = jax.vmap(val_from_im, (0, None))


def reproject_using_depth(uvs, depth_im, cam, intr):
    """
    Reprojects 2D key points from a given frame onto a second frame given
    a depth image and relative camera and intrinsics.

    Useful for creating keypoint position baselines if depth is available.
    """
    zs = vals_from_im(uvs, jnp.array(depth_im))
    xs = camera_from_screen_and_depth(uvs, zs, intr)
    valid = zs > 0
    uvs_ = screen_from_camera(cam.inv()(xs), intr)
    return uvs_, valid


def xq_dist(p, p_):
    """Returns the Euclidean distance between positions and between quaternions of two poses."""
    x0, x1 = p.pos, p_.pos
    q0, q1 = p.quat, p_.quat

    q0 = q0 / jnp.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / jnp.linalg.norm(q1, axis=-1, keepdims=True)

    xerr = jnp.linalg.norm(x0 - x1, axis=-1)
    qerr = jnp.minimum(
        jnp.linalg.norm(q0 - q1, axis=-1), jnp.linalg.norm(q0 + q1, axis=-1)
    )
    return xerr, qerr


def xq_cos(p, p_):
    """Returns the cosine similarity between positions and between quaternions of two poses."""
    x0, x1 = p.pos, p_.pos
    q0, q1 = p.quat, p_.quat

    q0 = q0 / jnp.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / jnp.linalg.norm(q1, axis=-1, keepdims=True)

    x0 = x0 / jnp.linalg.norm(x0, axis=-1, keepdims=True)
    x1 = x1 / jnp.linalg.norm(x1, axis=-1, keepdims=True)

    xerr = (x0 * x1).sum(axis=-1)
    qerr = jnp.maximum((q0 * q1).sum(axis=-1), (q0 * (-q1)).sum(axis=-1))
    return xerr, qerr


def slicify(im, shape):
    """
    Returns sliding windows of a given shape over an image.
    """
    slices_shape = (im.shape[0] - shape[0], im.shape[1] - shape[1])
    inds = jnp.indices(slices_shape + (1,)).reshape(3, -1).T
    slices = jax.vmap(jax.lax.dynamic_slice, (None, 0, None))(im, inds, shape + (3,))
    return slices.reshape(slices_shape + shape + (3,))


def rgb_to_gray(rgb):
    v = jnp.array([0.2125, 0.7154, 0.0721])
    return jnp.array(jnp.dot(rgb[..., :3], v))
