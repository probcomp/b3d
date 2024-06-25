import jax
import jax.numpy as jnp
from b3d.utils import keysplit
from .core import Pose
import genjax


def uniform_samples_from_disc(key, N, d=3):
    _, keys = keysplit(key, 1, 2)
    r = jax.random.uniform(keys[0], (N, 1), minval=0, maxval=1)
    phi = jax.random.normal(keys[1], (N, d))
    phi = phi / jnp.linalg.norm(phi, axis=1, keepdims=True)
    disc = r ** (1 / d) * phi

    return disc


# TODO: Is this correct?
def unit_disc_to_sphere(x):
    """
    Maps uniform samples `x` from the unit n-disc to the n-sphere to
    uniform samples in the 3-sphere centered around `(0,...,0,1)`.
    """
    r = jnp.linalg.norm(x, axis=-1, keepdims=True)
    phi = x / r
    xs = jnp.concatenate(
        [jnp.sin(r * jnp.pi / 2) * phi, jnp.cos(r * jnp.pi / 2)], axis=1
    )
    return xs


def volume_of_3_ball(r):
    return 4 * jnp.pi / 3 * r**3


# TODO: Is this correct??
def volume_of_cap_around_north_pole(r):
    """
    Returns the volume of $S^3 \cap ( \{ \sqrt{x^2 + y^2 + z^2}=1 \} \times \R )$
    """
    return jnp.pi * (jnp.pi - (jnp.sin(2 * jnp.arccos(r)) + 2 * jnp.arccos(r)))


def uniform_samples_from_SE3_around_identity(key, N, rx=1.0, rq=1.0):
    """
    Returns N samples from SE(3) around the identity, where positions
    are uniformly sampled from a unit-disc of radius rx, and quaternions are uniformly
    sampled from an embedded disc or radius rq in the 3-sphere centered around (0,0,0,1).


    Example:
    ```
    from b3d.pose_utils import uniform_samples_from_SE3_around_identity
    import numpy as np
    import jax
    import viser


    server = viser.ViserServer()
    key = jax.random.PRNGKey(0)

    ps  = uniform_samples_from_SE3_around_identity(
                key, N=20, rx=0.2, rq=0.1)

    for i in range(ps.pos.shape[0]):
        p = ps[i]
        server.add_frame(
            f"p[{i}]",
            position=np.array(p.pos),
            wxyz=np.array(p.wxyz),
            axes_length=0.2,
            axes_radius=0.01)
    ```
    """
    # TODO: assert rq <= 1.0, "rq should be <= 1"
    _, keys = keysplit(key, 1, 2)

    xs = rx * uniform_samples_from_disc(keys[0], N, d=3)
    qs = unit_disc_to_sphere(rq * uniform_samples_from_disc(keys[1], N, d=3))
    return Pose(xs, qs)


# class UniformPoseInBall(genjax.ExactDensity, genjax.JAXGenerativeFunction):
#     def sample(self, key, p0: Pose, rx, rq):
#         p1 = uniform_samples_from_SE3_around_identity(key, 1, rx, rq)[0]
#         return p0.compose(p1)

#     def logpdf(self, p, p0: Pose, rx, rq):
#         # TODO: Check if this is correct
#         # TODO: Check if p1 is within the bounds of the discs,
#         #       where `p1 = p0.inv().compose(p)`
#         return -jnp.log(volume_of_3_ball(rx)) - jnp.log(
#             volume_of_cap_around_north_pole(rq)
#         )

# uniform_pose_in_ball = UniformPoseInBall()
