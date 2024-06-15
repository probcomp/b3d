import jax
import jax.numpy as jnp
import genjax
from b3d.pose import Pose
from b3d.gps_utils import cov_from_dq_composition
from hgps.dynamic_gps import DynamicGPS
from typing import Any, TypeAlias


# TODO: Test this code
# NOTE: I'm adding T,N,K as params here because at some point genjax had trouble handling ints
#       because we don't have the equivalent of jax's static arguments yet.
#       (I Have to check if that's still the case.)
def make_sparse_gps_model(
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
    camera_motion_model_args
):
    """
    Models independently moving rigid object as clusters of
    Gaussians which specify the position, pose, and uncertainty of 3d keypoints in space.

    For simplicity we assume that keypoints can only emit a single constant feature,
    that we may interpret as a "point light".

    We can easily extend this model to handle feature vectors.

    Args:
        `T`: Number of time steps
        `N`: Number of particles
        `K`: Number of object clusters
        `F`: Feature dimension
        `particle_pose_prior`: Particle pose model `(nums: Array, *args) -> poses`.
        `particle_pose_prior_args`: Arguments for the particle pose prior
        `object_pose_prior`: Cluster prior model `(nums: Array, *args) -> poses`.
        `object_pose_prior_args`: Arguments for the object cluster pose prior
        `camera_pose_prior`: Camera pose prior model `(*args) -> pose`.
        `camera_pose_prior_args`: Arguments for the camera pose prior.
        `observation_model`: Observation model `(nums: Array: Array, mus, covs, cam, *args) -> observations`.
        `observation_model_args`: Arguments for the observation model
        `object_motion_model`: Object motion model `(poses, *args) -> poses`.
        `object_motion_model_args`: Arguments for the object motion model
        `camera_motion_model`: Camera motion model `(pose, *args) -> pose`.
        `camera_motion_model_args`: Arguments for the camera motion model
    """

    @genjax.unfold_combinator(max_length=T - 1)
    @genjax.static_gen_fn
    def unfolded_kernel(
        state, object_assignments, relative_particle_poses, particle_jitter, intr
    ):
        """Kernel modelling the time evolution a scene."""
        t, object_poses, camera_pose = state

        # New Cluster pose in world coordinates
        # TODO: should empty clusters be masked out?
        new_object_poses = (
            object_motion_model(object_poses, *object_motion_model_args)
            @ "object_poses"
        )

        # Absolute particle poses in world coordinates
        absolute_particle_poses = new_object_poses[object_assignments].compose(
            relative_particle_poses
        )
        particle_positions = absolute_particle_poses.pos
        particle_covariances = jax.vmap(cov_from_dq_composition)(
            particle_jitter, absolute_particle_poses.quat
        )

        # Updated camera pose in world coordinates
        new_camera_pose = (
            camera_motion_model(camera_pose, *camera_motion_model_args) @ "camera_pose"
        )

        _observed_2d_keypoints = (
            observation_model(
                vis_mask,
                particle_poses,
                camera_pose,
                intr,
                *observation_model_args,
            )
            @ "observation"
        )

        return (t + 1, new_object_poses, new_camera_pose)

    # TODO: What arguments should be passed to the model?
    @genjax.static_gen_fn
    def sparse_gps_model(intr):
        # Gaussian particles describing feature distributions over regions in space.
        # (Note that for simplicity we assume a unique point light feature.)
        # We describe their pose in the local coordinate system of the object (or cluster) they belong to.
        relative_particle_poses = (
            particle_pose_prior(jnp.arange(N), *particle_pose_prior_args)
            @ "particle_poses"
        )

        # Particle jitter encodes uncertainty of the particles 3d position as
        # a diagonal covariance matrix for which we sample the diagonal entries here.
        # TODO: Should we sample the jitter from a different distribution?
        particle_jitter = (
            genjax.normal(jnp.zeros((N, 3)), jitter_sigma * jnp.ones((N, 3)))
            @ "particle_jitter"
        )
        particle_jitter = particle_jitter**2

        # NOTE: Not used at the moment
        # TODO: Incorporate particle features into observation model
        _particle_features = (
            genjax.normal(jnp.zeros((N, F)), jnp.ones((N, F))) @ "particle_features"
        )

        # Each particle is associated with an object cluster
        # TODO: How should we handle empty clusters?
        object_assignments = (
            genjax.map_combinator(in_axes=(0,))(genjax.categorical)(jnp.zeros((N, K)))
            @ "object_assignments"
        )

        # Cluster pose in world coordinates
        initial_object_poses = (
            object_pose_prior(jnp.arange(K), *object_pose_prior_args)
            @ "initial_object_poses"
        )

        # Absolute particle poses in world coordinates
        absolute_particle_poses = initial_object_poses[object_assignments].compose(
            relative_particle_poses
        )
        particle_positions = absolute_particle_poses.pos
        particle_covariances = jax.vmap(cov_from_dq_composition)(
            particle_jitter, absolute_particle_poses.quat
        )

        # Initial camera pose in world coordinates
        initial_camera_pose = (
            camera_pose_prior(*camera_pose_prior_args) @ "initial_camera_pose"
        )

        _observed_2d_particles = (
            observation_model(
                jnp.arange(N),
                particle_positions,
                particle_covariances,
                initial_camera_pose,
                intr,
                *observation_model_args,
            )
            @ "initial_observation"
        )

        state0 = (0, initial_object_poses, initial_camera_pose)
        states = (
            unfolded_kernel(
                T - 1,
                state0,
                object_assignments,
                relative_particle_poses,
                particle_jitter,
                intr,
            )
            @ "chain"
        )

        object_poses = Pose(
            jnp.concatenate([state0[1][None, :].pos, states[1].pos], axis=0),
            jnp.concatenate([state0[1][None, :].quat, states[1].quat], axis=0),
        )

        camera_poses = Pose(
            jnp.concatenate([state0[2][None, :].pos, states[2].pos], axis=0),
            jnp.concatenate([state0[2][None, :].quat, states[2].quat], axis=0),
        )

        absolute_particle_poses = (
            object_poses[:, object_assignments] @ relative_particle_poses[None]
        )
        # TODO: Wrap this into a DynamicHGPS class.
        return {
            "relative_particle_poses": relative_particle_poses,
            "absolute_particle_poses": absolute_particle_poses,
            "object_poses": object_poses,
            "object_assignments": object_assignments,
            "camera_poses": camera_poses,
        }

    return sparse_gps_model


# # # # # # # # # # # # # # # # # # # # # #
#
#   Quick access utils
#
# # # # # # # # # # # # # # # # # # # # # #
SparseGPSModelTrace: TypeAlias = Any


def get_inner_val(tr, addr, *rest):
    """Hack till PR is merged."""
    inner_val = None
    if hasattr(tr[addr], "inner"):
        if hasattr(tr[addr].inner, "value"):
            inner_val = tr[addr].inner.value
        else:
            inner_val = tr[addr].inner
    else:
        inner_val = tr[addr]

    if len(rest) == 0:
        return inner_val
    else:
        return get_inner_val(inner_val, *rest)


def get_particles(tr: SparseGPSModelTrace):
    """Returns the particle poses, covariances, and features from a SparseGPSModelTrace."""
    return (
        get_inner_val(tr, "particle_poses"),
        tr["particle_jitter"],
        tr["particle_features"],
    )


def get_assignments(tr: SparseGPSModelTrace):
    return get_inner_val(tr, "object_assignments")


def get_object_poses(tr: SparseGPSModelTrace):
    return get_inner_val(tr, "initial_object_poses")[None, :].concat(
        get_inner_val(tr, "chain", "object_poses"), axis=0
    )


def get_cameras(tr: SparseGPSModelTrace):
    return get_inner_val(tr, "initial_camera_pose")[None, :].concat(
        get_inner_val(tr, "chain", "camera_pose")
    )


def get_2d_particle_positions(tr: SparseGPSModelTrace):
    return jnp.concatenate(
        [
            get_inner_val(tr, "initial_observation", "2d_particle_position")[None],
            get_inner_val(tr, "chain", "observation", "2d_particle_position"),
        ],
        axis=0,
    )


def get_dynamic_gps(tr: SparseGPSModelTrace):
    """Gets the DynamicGPS object from a SparseGPSModelTrace."""
    return DynamicGPS.from_pose_data(
        *get_particles(tr), get_assignments(tr), get_object_poses(tr)
    )


# # # # # # # # # # # # # # # # # # # # # #
#
#   Pre-configured model factory
#
# # # # # # # # # # # # # # # # # # # # # #
