from b3d.pose import Pose, Rot
import jax.numpy as jnp


def contact_parameters_to_pose(cp):
    return Pose(
        jnp.array([cp[0], cp[1], 0.0]),
        Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat(),
    )
