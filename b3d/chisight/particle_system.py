import jax.numpy as jnp
import b3d
from b3d import Pose
import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from b3d import Pose, Mesh
from b3d.chisight.sparse.gps_utils import add_dummy_var
from b3d.pose import uniform_pose_in_ball
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
    ), final_state

@genjax.gen
def sparse_observation_model(particle_absolute_poses, camera_pose, visibility, instrinsics, sigma):
    # TODO: add visibility
    uv = b3d.camera.screen_from_world(particle_absolute_poses.pos, camera_pose, instrinsics.const)
    uv_ = genjax.normal(uv, jnp.tile(sigma, uv.shape)) @ "sensor_coordinates"
    return uv_

@genjax.gen
def sparse_gps_model(latent_particle_model_args, obs_model_args):
    # (b3d.camera.Intrinsics.from_array(jnp.array([1.0, 1.0, 1.0, 1.0])), 0.1)
    particle_dynamics_summary, final_state = latent_particle_model(*latent_particle_model_args) @ "particle_dynamics"
    obs = sparse_observation_model.vmap(in_axes=(0, 0, 0, None, None))(
        particle_dynamics_summary["absolute_particle_poses"],
        particle_dynamics_summary["camera_pose"],
        particle_dynamics_summary["vis_mask"],
        *obs_model_args
    ) @ "obs"
    return (particle_dynamics_summary, final_state, obs)



def make_dense_gps_model(likelihood):
    """
    The provided likelihood should be a Generative Function with
    one latent choice at address `"image"`, which accepts `(mesh, other_args)` as input,
    and outputs `(image, metadata)`.
    The value `image` should be sampled at `"image"`.
    """

    @genjax.gen
    def dense_gps_model(latent_particle_model_args, dense_model_args):
        particle_dynamics_summary, final_state = latent_particle_model(*latent_particle_model_args) @ "particle_dynamics"
        absolute_particle_poses_last_frame = particle_dynamics_summary["absolute_particle_poses"][-1]
        camera_pose_last_frame = particle_dynamics_summary["camera_pose"][-1]
        absolute_particle_poses_in_camera_frame = camera_pose_last_frame.inv() @ absolute_particle_poses_last_frame
        
        batched_mesh, likelihood_args = dense_model_args
        merged_mesh = Mesh.transform_and_merge_meshes(batched_mesh, absolute_particle_poses_in_camera_frame)
        image = likelihood(merged_mesh, likelihood_args) @ "obs"
        return (particle_dynamics_summary, final_state, image)

    return dense_gps_model


def visualize_particle_system(latent_particle_model_args, particle_dynamics_summary, final_state):
    import rerun as rr
    (dynamic_state, static_state) = final_state

    (
        num_timesteps, # const object
        num_particles, # const object
        num_clusters, # const object
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params
    ) = latent_particle_model_args

    colors = b3d.distinct_colors(num_clusters.const)
    absolute_particle_poses = particle_dynamics_summary["absolute_particle_poses"]
    object_poses = particle_dynamics_summary["object_poses"]
    camera_pose = particle_dynamics_summary["camera_pose"]
    object_assignments = static_state[0]

    cluster_colors = jnp.array(b3d.distinct_colors(num_clusters.const))

    for t in range(num_timesteps.const):
        rr.set_time_sequence("time", t)

        cam_pose = camera_pose[t]
        rr.log(
            f"/camera",
            rr.Transform3D(translation=cam_pose.position, rotation=rr.Quaternion(xyzw=cam_pose.xyzw)),
        )
        rr.log(
            f"/camera",
            rr.Pinhole(
                resolution=[0.1,0.1],
                focal_length=0.1,
            ),
        )

        rr.log(
            "absolute_particle_poses",
            rr.Points3D(
                absolute_particle_poses[t].pos,
                colors=cluster_colors[object_assignments]
            )
        )

        for i in range(num_clusters.const):
            b3d.rr_log_pose(f"cluster/{i}", object_poses[t][i])

def particle_2d_pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)].set(jnp.arange(len(pixel_coords))+1 )
    return img

def visualize_sparse_observation(sparse_model_args, observations):
    import rerun as rr
    intrinsics = sparse_model_args[0].const

    for t in range(observations.shape[0]):
        rr.set_time_sequence("time", t)
        img = particle_2d_pixel_coordinates_to_image(observations[t], intrinsics.height, intrinsics.width)
        rr.log(
            "obs",
            rr.DepthImage(img)
        )

def visualize_dense_gps(latent_particle_model_args, dense_model_args, particle_dynamics_summary, final_state):

    (
        num_timesteps, # const object
        num_particles, # const object
        num_clusters, # const object
        relative_particle_poses_prior_params,
        initial_object_poses_prior_params,
        camera_pose_prior_params
    ) = latent_particle_model_args
    (meshes, _) = dense_model_args

    import rerun as rr
    for i in range(len(meshes)):
        rr.log(f"/particle_meshes/{i}",
            rr.Mesh3D(
                vertex_positions=meshes[i].vertices,
                triangle_indices=meshes[i].faces,
                vertex_colors=meshes[i].vertex_attributes), timeless=True)

    for t in range(num_timesteps.const):
        rr.set_time_sequence("time", t)
        poses = particle_dynamics_summary["absolute_particle_poses"][t]
        for i in range(len(meshes)):
            pose = poses[i]
            rr.log(
                f"/particle_meshes/{i}",
                rr.Transform3D(translation=pose.position, rotation=rr.Quaternion(xyzw=pose.xyzw)),
            )

