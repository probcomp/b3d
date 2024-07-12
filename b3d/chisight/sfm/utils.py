import jax
import jax.numpy as jnp
from b3d.camera import screen_from_world
from b3d.pose import Pose, Rot


def reprojection_error(xs, us, cam, intr):
    us_ = screen_from_world(xs, cam, intr)
    err = jnp.linalg.norm(us_ - us, axis=-1).sum()
    return err


def line_intersects_box(x, dx, width, height):
    

    dx = dx/jnp.linalg.norm(dx)

    # Define the box boundaries
    x_min, x_max = 0.0, width
    y_min, y_max = 0.0, height

    # Define the parameter t for the line equation x + t * dx
    # t0_x = jnp.where(dx[0] != 0.0, (x_min - x[0]) / dx[0], -jnp.inf)
    # t1_x = jnp.where(dx[0] != 0.0, (x_max - x[0]) / dx[0], jnp.inf)
    # t0_y = jnp.where(dx[1] != 0.0, (y_min - x[1]) / dx[1], -jnp.inf)
    # t1_y = jnp.where(dx[1] != 0.0, (y_max - x[1]) / dx[1], jnp.inf)
    t0_x = (x_min - x[0]) / dx[0]
    t1_x = (x_max - x[0]) / dx[0]
    t0_y = (y_min - x[1]) / dx[1]
    t1_y = (y_max - x[1]) / dx[1]

    ps = jnp.array([
        x + t0_x * dx,
        x + t1_x * dx,
        x + t0_y * dx,
        x + t1_y * dx
    ])

    eps=1e-3
    valid = (
        (-eps <= ps[:,0]) *
        (ps[:,0] <= width+eps) * 
        (-eps <= ps[:,1]) * 
        (ps[:,1] <= height+eps)
    )

    # TODO: Fix edgecases when x is far out
    ds = jnp.abs(ps - jnp.array([[width/2, height/2]])).sum(1)
    inds = jnp.argsort(ds)[:2]
    seg = jax.lax.cond(valid.sum()>=2, lambda: ps[inds], lambda: jnp.tile(-jnp.inf, (2,2)))


    return seg[0], seg[1]