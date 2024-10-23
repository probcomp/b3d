import jax
import jax.numpy as jnp

from b3d.camera import (
    camera_from_screen,
    camera_from_screen_and_depth,
    screen_from_world,
    world_from_screen,
)
from b3d.pose import uniform_pose_in_ball
from b3d.utils import keysplit


# # # # # # # # # # # # # # # # # # # #
#
#   Helper
#
# # # # # # # # # # # # # # # # # # # #
def dist_to_and_along_line(u, ell):
    """
    Returns the distance of 'u' to the line through 'l', and
    the amount that of `u` along `l/|l|`, that is,

        `|dot(u, il/|il|)|` and `dot(u, l/|l|)`.
    """
    # Normalize and
    # rotate by 90 degrees
    ell = ell / jnp.sqrt(ell[..., [0]] ** 2 + ell[..., [1]] ** 2)
    iell = jnp.stack([-ell[..., 1], ell[..., 0]], axis=-1)
    d = u[..., 0] * iell[..., 0] + u[..., 1] * iell[..., 1]
    s = u[..., 0] * ell[..., 0] + u[..., 1] * ell[..., 1]
    return jnp.abs(d), s


def distance_to_line(uv, ell):
    """
    Returns the distance of a 2D image point to a line.

    Args:
        uv: 2D image point
        ell: Line in 2D described by `ax + by + c = 0`
    """
    s = (uv[..., 0] * ell[..., 0] + uv[..., 1] * ell[..., 1] + ell[..., 2]) / (
        ell[..., 0] ** 2 + ell[..., 1] ** 2
    )
    return jnp.abs(s)


def epipolar_pixel_distance(E, y0, y1, K, K_inv):
    """
    Given Essential matrix `E`, and two matched points `y0` and `y1`
    in normalized image coordinates, return the distance of `y1` to
    the epipolar line induced by `y0` on the sensor canvas.

    Args:
        E: Essential matrix
        y0: Normalized image coordinates at time 0
        y1: Normalized image coordinates at time 1
        K: Intrinsics matrix
        K_inv: Inverse of intrinsics matrix
    """
    ell = K_inv.T @ E @ y0
    u = K @ y1 / y1[2]
    return distance_to_line(u[:2], ell)


def epipolar_pixel_distances(E, ys0, ys1, K, K_inv):
    """
    Given Essential matrix `E`, and a collection of matched points `ys0` and `ys1`
    in normalized image coordinates, return the distances of `ys1` to
    the epipolar lines induced by `ys0` on the sensor canvas.
    """
    return jax.vmap(epipolar_pixel_distance, (None, 0, 0, None, None))(
        E, ys0, ys1, K, K_inv
    )


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


def _epi_constraint(cam, u0, u1, intr):
    """
    Epipolar constraint, but phrased in terms of the relative camera pose.
    Computes the alignment between the epipolar planes
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
    v0 = v0 / jnp.sqrt(v0[..., [0]] ** 2 + v0[..., [1]] ** 2 + v0[..., [2]] ** 2)
    v1 = v1 / jnp.sqrt(v1[..., [0]] ** 2 + v1[..., [1]] ** 2 + v1[..., [2]] ** 2)
    c = c / jnp.sqrt(c[..., [0]] ** 2 + c[..., [1]] ** 2 + c[..., [2]] ** 2)

    # Normal vector of epi-plane
    n = jnp.cross(c[None], v0, axis=-1)
    n = n / jnp.sqrt(n[..., [0]] ** 2 + n[..., [1]] ** 2 + n[..., [2]] ** 2)

    # Angle between epi-plane normal vector
    # and target obseration
    d = (n * v1).sum(-1)
    h = jnp.abs(d)

    return h, None


vmap_epi_constraint = jax.vmap(
    lambda cam, uv0, uv1, intr: _epi_constraint(cam, uv0, uv1, intr)[0],
    (0, None, None, None),
)


def _angle_check(cam, u0, u1, intr):
    """
    Checks if hypothetic intersections would lie in front of both cameras.

    Args:
        cam: Relative camera Pose
        u0: Array of shape (..., 2)
        u1: Array of shape (..., 2) (same shape as `u0`)
        intr: Intrinsics

    Returns
        Array of bools of shape (...)
    """
    # TODO: Add a reference, and check if this makes sense!?
    # NOTE: We work with a relative pose here, that is, we assume
    #       at time 0 world and camera frames are the same.

    # Extract relevant epipolar data
    # and normalize
    v0 = camera_from_screen(u0, intr)
    v1 = world_from_screen(u1, cam, intr) - cam.pos
    c = cam.pos

    v0 = v0 / jnp.sqrt(v0[..., [0]] ** 2 + v0[..., [1]] ** 2 + v0[..., [2]] ** 2)
    v1 = v1 / jnp.sqrt(v1[..., [0]] ** 2 + v1[..., [1]] ** 2 + v1[..., [2]] ** 2)
    c = c / jnp.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)

    # We first compute the Normal of epipolar plane and use it to
    # rotate c by 90 degrees in plane spanned by v0 and c.
    n = jnp.cross(c[None], v0, axis=-1)
    n = n / jnp.sqrt(n[..., [0]] ** 2 + n[..., [1]] ** 2 + n[..., [2]] ** 2)

    ic = jnp.cross(n, c[None], axis=-1)
    ic = ic / jnp.sqrt(ic[..., [0]] ** 2 + ic[..., [1]] ** 2 + ic[..., [2]] ** 2)

    # Coordinates of v0 and v1 in plane spanned by c and ic,
    # and their angles.
    x0 = (v0 * c[None]).sum(-1)
    y0 = (v0 * ic).sum(-1)

    x1 = (v1 * c[None]).sum(-1)
    y1 = (v1 * ic).sum(-1)

    a0 = jnp.arctan2(y0, x0)
    a1 = jnp.arctan2(y1, x1)

    return (0 < a0) & (a0 < a1) & (a1 < jnp.pi)


vmap_angle_check = jax.vmap(
    lambda cam, uv0, uv1, intr: _angle_check(cam, uv0, uv1, intr), (0, None, None, None)
)


def _ortho_score(cam, u0, u1, intr):
    """
    Computes a score measuring if keypoint lines are orthogonal --
    "zero" being co-linear, and "one" being orthogonal.

    This may be used as a heuristic scoring uncertainty of
    the depth estimate for each keypoint from the given pose.

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

    # Normalize
    v0 = v0 / jnp.sqrt(v0[..., [0]] ** 2 + v0[..., [1]] ** 2 + v0[..., [2]] ** 2)
    v1 = v1 / jnp.sqrt(v1[..., [0]] ** 2 + v1[..., [1]] ** 2 + v1[..., [2]] ** 2)

    v0_dot_v1 = (v0 * v1).sum(-1)

    return 1.0 - jnp.abs(v0_dot_v1)


vmap_ortho_score = jax.vmap(
    lambda cam, uv0, uv1, intr: _ortho_score(cam, uv0, uv1, intr), (0, None, None, None)
)


# NOTE: Experimental, don't rely on this
def _epi_constraint_variation_1(cam, u0, u1, intr):
    h, aux = _epi_constraint(cam, u0, u1, intr)[0]
    v0 = aux["v0"]
    v1_ = aux["v1_on_epiplane"]
    c = cam.pos / jnp.linalg.norm(cam.pos)
    h = h - jnp.sign((v0 * c).sum(-1) - (v1_ * c).sum(-1)).sum()
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
    ell = v1 - e
    u = u1 - e

    d, _ = dist_to_and_along_line(u, ell)
    aux = {
        "epipole": e,
        "line_direction": ell,
    }
    return d, aux


vmap_epi_distance = jax.vmap(
    lambda cam, uv0, uv1, intr: _epi_distance(cam, uv0, uv1, intr)[0],
    (0, None, None, None),
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
    ell = v1 - e
    u = u1 - e
    d, s = dist_to_and_along_line(u, ell)

    l_norm = jnp.sqrt(ell[..., [0]] ** 2 + ell[..., [1]] ** 2)

    proj_vec = s[..., None] * ell / l_norm
    error_vec = proj_vec - u

    x_near = camera_from_screen_and_depth(u0, jnp.array([intr.near]), intr)
    x_far = camera_from_screen_and_depth(u0, jnp.array([intr.far]), intr)
    v_near = screen_from_world(x_near, cam, intr)
    v_far = screen_from_world(x_far, cam, intr)
    vs = jnp.stack([v_near, v_far], axis=1)

    v0 = camera_from_screen(u0, intr)
    v1 = world_from_screen(u1, cam, intr) - cam.pos
    c = cam.pos

    # Normalize
    v0 = v0 / jnp.sqrt(v0[..., [0]] ** 2 + v0[..., [1]] ** 2 + v0[..., [2]] ** 2)
    v1 = v1 / jnp.sqrt(v1[..., [0]] ** 2 + v1[..., [1]] ** 2 + v1[..., [2]] ** 2)
    c = c / jnp.sqrt(c[..., [0]] ** 2 + c[..., [1]] ** 2 + c[..., [2]] ** 2)
    n = jnp.cross(v0, c[None], axis=-1)
    n = n / jnp.sqrt(n[..., [0]] ** 2 + n[..., [1]] ** 2 + n[..., [2]] ** 2)

    return dict(
        epipole=e,
        line_directions=ell,
        epi_distance=d,
        epi_scalar=s,
        projection_vector=proj_vec,
        error_vector=error_vec,
        near_far_screen=vs,
        near_far_world=jnp.stack([x_near, x_far], axis=1),
        v0=v0,
        v1=v1,
        c=c,
        n=jnp.cross(v0, c[None], axis=-1),
    )


# # # # # # # # # # # # # # # # # # # #
#
#   Helper
#
# # # # # # # # # # # # # # # # # # # #


def angle(v, w):
    v = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    w = w / jnp.linalg.norm(w, axis=-1, keepdims=True)
    return jnp.arccos((v * w).sum(-1))


# # # # # # # # # # # # # # # # # # # #
#
#   Proposal Factories
#
# # # # # # # # # # # # # # # # # # # #
vmap_uniform_pose = jax.jit(
    jax.vmap(uniform_pose_in_ball.sample, (0, None, None, None))
)


def make_two_frame_proposal(loss_func, choose_winner=jnp.argmin):
    """
    Returns a pose proposal, using the following recipe.
        - Sample *uniformly* around target pose, then
        - compute the lossess, and
        - return the the argmin.

    Args:
        loss_func: Function that takes `(q, uvs0, uvs1, intr)` and returns `(losses_per_keypoint, aux)`.
        choose_winner: Function that takes `losses_per_pose` and returns a winner index `i`.
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
        _, key = jax.random.split(key)
        keys = jax.random.split(key, S)
        qs = vmap_uniform_pose(keys, q, rx, rq)
        losses_ = jax.vmap(loss_func, (0, None, None, None))(qs, uvs0, uvs1, intr)[0]
        losses = jnp.nan_to_num(losses_.sum(1), nan=jnp.inf)

        # Pick best test pose
        # TODO: Resample?
        i = choose_winner(losses)
        q = qs[i]

        aux = {
            "proposals": p0 @ qs,
            "loss": losses,
            "winner_index": i,
            "winner_loss": losses[i],
        }

        return p0 @ q, aux

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

    x0 = camera_from_screen_and_depth(u0, intr.far * jnp.ones(u0.shape[:-1]), intr)
    ell = screen_from_world(x0, cam, intr)
    l_norm = jnp.sqrt(ell[..., 0] ** 2 + ell[..., 1] ** 2)

    ell = ell - e
    u = u1 - e

    # TODO: Constrain so that we only consider the
    #   positive part of the line. That "should" (might) get rid of
    #   weird local maxima with points behind the camera.
    d, s = dist_to_and_along_line(u, ell)
    d = jnp.where(s > 0.0, d, 1e2)
    d = jnp.where(s < l_norm, d, 1e2)

    s = jnp.clip(s, 0.0, jnp.inf)
    ys = e + s[:, None] * ell / l_norm[:, None]

    return d, ys
