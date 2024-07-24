import jax

from b3d.camera import Intrinsics
from b3d.chisight.sparse.gps_utils import add_dummy_var
from b3d.chisight.sparse.sparse_gps_model import (
    make_sparse_gps_model,
    minimal_observation_model,
)
from b3d.pose import Pose
from b3d.pose.pose_utils import uniform_pose_in_ball
from b3d.utils import keysplit

key = jax.random.PRNGKey(0)
dummy_mapped_uniform_pose = add_dummy_var(uniform_pose_in_ball).vmap(
    in_axes=(0, None, None, None)
)


intr = Intrinsics(100, 50, 100.0, 100.0, 50.0, 25.0, 1e-6, 100.0)
outlier_prob = 0.0

p0 = Pose.identity()
particle_pose_prior = dummy_mapped_uniform_pose
particle_pose_prior_args = (p0, 0.5, 0.25)

object_pose_prior = dummy_mapped_uniform_pose
object_pose_prior_args = (p0, 2.0, 0.5)

camera_pose_prior = uniform_pose_in_ball
camera_pose_prior_args = (p0, 0.1, 0.1)

observation_model = minimal_observation_model
observation_model_args = (2.0,)

object_motion_model = uniform_pose_in_ball.vmap(in_axes=(0, None, None))
object_motion_model_args = (0.1, 0.1)

camera_motion_model = uniform_pose_in_ball
camera_motion_model_args = (0.1, 0.2)

T, N, K = 2, 3, 3
F = 0
maker_args = (
    T,
    N,
    K,
    F,
    particle_pose_prior,
    particle_pose_prior_args,
    object_pose_prior,
    object_pose_prior_args,
    camera_pose_prior,
    camera_pose_prior_args,
    observation_model,
    observation_model_args,
    object_motion_model,
    object_motion_model_args,
    camera_motion_model,
    camera_motion_model_args,
)
model = make_sparse_gps_model(*maker_args)
jimportance = jax.jit(model.importance)
jsimulate = jax.jit(model.simulate)


key = keysplit(key)
tr = jsimulate(key, (intr,))
