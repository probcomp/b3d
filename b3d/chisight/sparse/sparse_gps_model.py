import jax
import jax.numpy as jnp
import genjax
from b3d.pose import Pose
from .dynamic_gps import DynamicGPS
from typing import Any, TypeAlias
from b3d.camera import screen_from_world


@genjax.gen
def minimal_observation_model(
        vis,
        particle_poses,
        camera_pose,
        intr,
        sigma
    ):
    """Simple observation model for debugging."""
    uv = screen_from_world(particle_poses.pos, camera_pose, intr)
    uv_ = genjax.normal(uv, jnp.tile(sigma, uv.shape)) @ "sensor_coordinates"
    return uv_


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
        `observation_model`: Observation model `(vis: Array: Array, mus, cam, intr, *args) -> observations`.
        `observation_model_args`: Arguments for the observation model
        `object_motion_model`: Object motion model `(poses, *args) -> poses`.
        `object_motion_model_args`: Arguments for the object motion model
        `camera_motion_model`: Camera motion model `(pose, *args) -> pose`.
        `camera_motion_model_args`: Arguments for the camera motion model
    """
    @genjax.gen
    def kernel(carried_state, _):
        """Kernel modelling the time evolution a scene."""
        t, object_poses, camera_pose, static_carries = carried_state
        (object_assignments, relative_particle_poses, intr) = static_carries

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

        # Updated camera pose in world coordinates
        new_camera_pose = (
            camera_motion_model(camera_pose, *camera_motion_model_args) 
            @ "camera_pose"
        )

        # Visibility mask
        vis_mask = (
            genjax.bernoulli.vmap(in_axes=(0,))(jnp.repeat(jax.scipy.special.logit(0.5), N))
            @ "visibility"
        )
        
        _observation = (
            observation_model(
                vis_mask,
                absolute_particle_poses,
                new_camera_pose,
                intr,
                *observation_model_args,
            )
            @ "observation"
        )

        new_state = (t + 1, new_object_poses, new_camera_pose, static_carries)
        vector_output = None
        return (new_state, new_state)

    # TODO: What arguments should be passed to the model?
    @genjax.gen
    def sparse_gps_model(intr):
        # Gaussian particles describing feature distributions over regions in space.
        # (Note that for simplicity we assume a unique point light feature.)
        # We describe their pose in the local coordinate system of the object (or cluster) they belong to.
        relative_particle_poses = (
            particle_pose_prior(jnp.arange(N), *particle_pose_prior_args)
            @ "particle_poses"
        )

        # Each particle is associated with an object cluster
        # NOTE: this could also vary for each partile
        # TODO: How should we handle empty clusters?
        object_assignments = (
            genjax.categorical.vmap(in_axes=(0,))(jnp.zeros((N, K)))
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

        # Initial camera pose in world coordinates
        initial_camera_pose = (
            camera_pose_prior(*camera_pose_prior_args) 
            @ "initial_camera_pose"
        )

        # TODO: Make the visibility model an argument to the model factory
        initial_vis_mask = (
            genjax.bernoulli.vmap(in_axes=(0,))(
                jnp.repeat(jax.scipy.special.logit(0.5), N))
            @ "initial_visibility"
        )

        _initial_observation = (
            observation_model(
                initial_vis_mask,
                absolute_particle_poses,
                initial_camera_pose,
                intr,
                *observation_model_args,
            )
            @ "observation"
        )

        static_carried_values = (object_assignments, relative_particle_poses, intr)
        state0 = (0, initial_object_poses, initial_camera_pose, static_carried_values)
        last_state, ret = kernel.scan(n=T-1)(state0, None) @ "chain"

        # Combine the initial state with the chain of states
        object_poses = Pose(
            jnp.concatenate([state0[1][None, :].pos, ret[1].pos], axis=0),
            jnp.concatenate([state0[1][None, :].quat, ret[1].quat], axis=0),
        )

        camera_poses = Pose(
            jnp.concatenate([state0[2][None, :].pos, ret[2].pos], axis=0),
            jnp.concatenate([state0[2][None, :].quat, ret[2].quat], axis=0),
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

def get_particle_poses(tr: SparseGPSModelTrace):
    # TODO: is there a better way to do this?
    return tr.get_choices()("particle_poses").c.v

def get_assignments(tr: SparseGPSModelTrace):
    # TODO: is there a better way to do this?
    return tr.get_choices()("object_assignments").c.v


def get_object_poses(tr: SparseGPSModelTrace):
    # TODO
    pass

def get_cameras(tr: SparseGPSModelTrace):
    # TODO
    pass


def get_2d_particle_positions(tr: SparseGPSModelTrace):
    # TODO
    pass


def get_dynamic_gps(tr: SparseGPSModelTrace):
    """Gets the DynamicGPS object from a SparseGPSModelTrace."""
    ps = get_particle_poses(tr)
    qs = get_object_poses(tr)

    T = qs.shape[0]
    N = ps.shape[0]

    pos  = jnp.tile(ps.pos, (T,1,1))
    quat = jnp.tile(ps.quat, (T,1,1))
    ps = Pose(pos, quat)

    diag = jnp.ones((N, 3))

    return DynamicGPS.from_pose_data(
        ps, diag, jnp.zeros((N,1)), get_assignments(tr), qs
    )


# # # # # # # # # # # # # # # # # # # # # #
#
#   Pre-configured model factory
#
# # # # # # # # # # # # # # # # # # # # # #



