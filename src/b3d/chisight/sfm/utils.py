import jax

from b3d.pose import uniform_pose_in_ball

vmap_uniform_pose = jax.jit(jax.vmap(uniform_pose_in_ball.sample, (0,None,None,None)))

uniform_quat_samples_around_identity(key, N, rq)