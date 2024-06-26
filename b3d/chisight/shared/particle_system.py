import jax.numpy as jnp
import b3d
from b3d import Pose
import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from b3d.chisight.dense.dense_likelihood import make_dense_image_likelihood_from_renderer, DenseImageLikelihoodArgs

# Initial idea [we changed below]-
# Factory Function(observation_model)
# - object poses
# - assignments
# - particle relative poses
# - camera_poses 
# - visibility
# - observation_model(particle_absolute_poses, camera_poses, visibility)

from b3d.chisight.sparse.gps_utils import add_dummy_var
from b3d.chisight.sparse.pose_utils import uniform_pose_in_ball
dummy_mapped_uniform_pose = add_dummy_var(uniform_pose_in_ball).vmap(in_axes=(0,None,None,None))


uniform_pose_args = (Pose.identity(), 2.0, 0.5)

@gen
def initial_particle_system_state(
    num_particles,
    num_clusters,
    relative_particle_poses_prior_params,
    initial_object_poses_prior_params,
    camera_pose_prior_params,
):
    relative_particle_poses = (
        dummy_mapped_uniform_pose(jnp.arange(num_particles.const), *relative_particle_poses_prior_params)
        @ "particle_poses"
    )

    object_assignments = (
        genjax.categorical.vmap(in_axes=(0,))(jnp.zeros((num_particles.const, num_clusters.const)))
        @ "object_assignments"
    )

    # Cluster pose in world coordinates
    initial_object_poses = (
        dummy_mapped_uniform_pose(jnp.arange(num_clusters.const), *initial_object_poses_prior_params)
        @ "object_poses"
    )

    # Absolute particle poses in world coordinates
    absolute_particle_poses = initial_object_poses[object_assignments].compose(
        relative_particle_poses
    )

    # Initial camera pose in world coordinates
    initial_camera_pose = (
        uniform_pose_in_ball(*camera_pose_prior_params) 
        @ "initial_camera_pose"
    )

    # Initial visibility mask
    initial_vis_mask = (
        genjax.bernoulli.vmap(in_axes=(0,))(
            jnp.repeat(jax.scipy.special.logit(0.5), num_particles.const))
        @ "initial_visibility"
    )

    dynamic_state = (initial_object_poses, initial_camera_pose)
    static_state = (object_assignments, relative_particle_poses, num_particles)

    return (
        (dynamic_state, static_state),
        {
            "relative_particle_poses": relative_particle_poses,
            "absolute_particle_poses": absolute_particle_poses,
            "object_poses": initial_object_poses,
            "camera_pose": initial_camera_pose,
            "vis_mask": initial_vis_mask
        }
    )

@gen
def particle_system_state_step(carried_state, _):
    dynamic_state, static_state = carried_state
    object_poses, camera_pose = dynamic_state
    object_assignments, relative_particle_poses, num_particles = static_state

    new_object_poses = (
        uniform_pose_in_ball.vmap(in_axes=(0, None, None))(object_poses, 0.1, 0.1)
        @ "object_poses"
    )

    # Absolute particle poses in world coordinates
    absolute_particle_poses = new_object_poses[object_assignments].compose(
        relative_particle_poses
    )

    # Updated camera pose in world coordinates
    new_camera_pose = (
        uniform_pose_in_ball(camera_pose, 0.1, 0.2) 
        @ "camera_pose"
    )

    # Visibility mask
    vis_mask = (
        genjax.bernoulli.vmap(in_axes=(0,))(jnp.repeat(jax.scipy.special.logit(0.5), num_particles.const))
        @ "visibility"
    )

    new_dynamic_state = (new_object_poses, new_camera_pose)
    new_state = (new_dynamic_state, static_state)
    return (
        new_state,
        {
            "relative_particle_poses": relative_particle_poses,
            "absolute_particle_poses": absolute_particle_poses,
            "object_poses": new_object_poses,
            "camera_pose": new_camera_pose,
            "vis_mask": vis_mask
        }
    )

@gen
def latent_particle_model(
        num_timesteps, # const object
        num_particles, # const object
        num_clusters, # const object
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params
    ):
    """
    Retval is a dict with keys "relative_particle_poses", "absolute_particle_poses",
    "object_poses", "camera_poses", "vis_mask"
    Leading dimension for each timestep is the batch dimension.
    """
    (state0, init_retval) = initial_particle_system_state(
        num_particles, num_clusters,
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params
    ) @ "state0"

    final_state, scan_retvals = particle_system_state_step.scan(n=(num_timesteps.const - 1))(state0, None) @ "states1+"

    # concatenate each element of init_retval, scan_retvals
    return jax.tree.map(
        lambda t1, t2: jnp.concatenate([t1[None, :], t2], axis=0),
        init_retval, scan_retvals
    )

@genjax.gen
def sparse_observation_model(particle_absolute_poses, camera_pose, visibility, instrinsics, sigma):
    # TODO: add visibility
    uv = b3d.camera.screen_from_world(particle_absolute_poses.pos, camera_pose, instrinsics.const)
    uv_ = genjax.normal(uv, jnp.tile(sigma, uv.shape)) @ "image"
    return uv_

@genjax.gen
def sparse_gps_model(latent_particle_model_args, obs_model_args):
    # (b3d.camera.Intrinsics.from_array(jnp.array([1.0, 1.0, 1.0, 1.0])), 0.1)
    particle_dynamics_summary = latent_particle_model(*latent_particle_model_args) @ "particle_dynamics"
    obs = sparse_observation_model.vmap(in_axes=(0, 0, 0, None, None))(
        particle_dynamics_summary["absolute_particle_poses"],
        particle_dynamics_summary["camera_pose"],
        particle_dynamics_summary["vis_mask"],
        *obs_model_args
    ) @ "obs"
    return (particle_dynamics_summary, obs)


def dense_gps_model_factory(dense_image_likelihood):

    @genjax.static_gen_fn
    def dense_gps_model(
        meshes,
        dense_likelihood_args,
        *latent_particle_model_args
    ):
        particle_dynamics_summary = latent_particle_model(*latent_particle_model_args)
        absolute_particle_poses_last_frame = particle_dynamics_summary["absolute_particle_poses"][-1]
        camera_pose_last_frame = particle_dynamics_summary["camera_poses"][-1]

        absolute_particle_poses_in_camera_frame = camera_pose_last_frame.inv() @ absolute_particle_poses_last_frame

        image = dense_image_likelihood(absolute_particle_poses_in_camera_frame, meshes, dense_likelihood_args) @ "image"
        return image

    return dense_gps_model


