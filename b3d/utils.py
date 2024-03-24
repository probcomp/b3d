import jax.numpy as jnp

def xyz_from_depth(
    z: "Depth Image", 
    fx, fy, cx, cy
):
    v, u = jnp.mgrid[: z.shape[0], : z.shape[1]] + 0.5
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz
