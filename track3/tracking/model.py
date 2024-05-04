import jax
import jax.numpy as jnp
import genjax
import rerun as rr
import b3d
from .dists import gaussian_vmf_2d
from .mesh import Mesh, rerun_mesh_rep

def model_factory(*,
        max_T,
        width,
        height,
        pose_kernel_params,
        max_keypoint_mesh_width,
        max_keypoint_mesh_height,
        max_depth
    ):
    """
    Args:
        - max_T : int
            Maximum number of frames
        - width : int
        - height: int
        - pose_kernel_params : (float, float)
            (variance, concentration) for pose kernel

    Notes:
        - 2D pixel (i, j) corresponds to the square with bottom left corner
            [i, j] and top right corner [i+1, j+1].
    """
    
    generate_keypoint_mesh = get_generate_keypoint_mesh(max_keypoint_mesh_width, max_keypoint_mesh_height, max_depth)
    pose_hmm = get_pose_hmm(max_T, pose_kernel_params, width, height)
    obs_generator = get_obs_model(width, height)

    @genjax.static_gen_fn
    def model_1keypoint_2d(T):
        """
        Simple model variant.
        - 2D
        - Keypoint begins at frame 1 and persists until frame T
        - Exactly 1 keypoint
        """
        keypoint_mesh = generate_keypoint_mesh() @ "keypoint_mesh"
        poses = pose_hmm(T) @ "poses"
        obs = obs_generator(poses, keypoint_mesh) @ "obs"
        return (poses, keypoint_mesh, obs)
    
    return model_1keypoint_2d

def get_pose_hmm(max_T, pose_kernel_params, width, height):
    # TODO: add a prior on the variance and concentration
    (variance, concentration) = pose_kernel_params
    
    @genjax.unfold_combinator(max_length=max_T)
    @genjax.static_gen_fn
    def unfold_poses(pose):
        new_pose = gaussian_vmf_2d(pose, variance, concentration) @ "pose"
        return new_pose

    @genjax.static_gen_fn
    def pose_hmm(T):
        """
        Args:
            - T : int
                Number of frames
        """
        low = jnp.array([width//2 + 0.01, width//2 + 0.1, 0.])
        high = jnp.array([width//2 + 0.0101, width//2 + 0.0101, 0.0001])
        # low = jnp.array([0., 0., 0.])
        # high = jnp.array([0 + 0.0001, 0 + 0.0001, 0.0001])
        initial_pose = genjax.uniform(low, high) @ "init"
        subsequent_poses = unfold_poses(T, initial_pose) @ "step"
        return jnp.concatenate([initial_pose[None, :], subsequent_poses])

    return pose_hmm

def get_generate_keypoint_mesh(maxwidth, maxheight, maxdepth):
    @genjax.static_gen_fn
    def generate_keypoint_mesh():
        lows = jnp.zeros((maxwidth, maxheight, 4))
        highs = jnp.ones((maxwidth, maxheight, 4)) * jnp.array([1., 1., 1., maxdepth])
        rgbd = genjax.map_combinator(in_axes=(1, 1))(
            genjax.map_combinator(in_axes=(0, 0))(genjax.uniform))(
            lows, highs
        ) @ "rgbd"
        return Mesh.mesh_from_pixels(maxwidth, maxheight, rgbd).scale(2)
    return generate_keypoint_mesh

def get_obs_model(width, height):
    DEFAULT_DEPTH = -1.

    @genjax.static_gen_fn
    def image_noise(image):
        return image
    
    @genjax.static_gen_fn
    def generate_background():
        return Mesh.square_mesh(
            jnp.array([0., 0.]),
            jnp.array([width, height]),
            jnp.array([1., 1., 1., 5.])
        )

    @genjax.map_combinator(in_axes=(0, None))
    @genjax.static_gen_fn
    def obs_model(keypoint_pose, keypoint_mesh : Mesh):
        deterministic_image = keypoint_mesh.to_image(
            keypoint_pose, width, height,
            lambda rgbd: rgbd[-1],
            jnp.array([0., 0., 0., DEFAULT_DEPTH])
        )
        observed_image = image_noise(deterministic_image) @ "observed_image"
        # observed_image = b3d.rgbd_sensor_model(
        #     deterministic_image[:, :, :3],
        #     deterministic_image[:, :, 3],
            
        # )
        return observed_image
    return obs_model

### Viz ###
def rerun_log_trace(trace):
    (poses, keypoint_mesh, obs) = trace.get_retval()
    T = trace.get_args()[0]
    for t in range(T):
        rr.set_time_sequence("frames", t)
        rr.log("/obs/rgb", rr.Image(obs[t, ..., :3]))
        rr.log("/obs/depth", rr.DepthImage(obs[t, ..., 3]))

        kp_mesh = keypoint_mesh.transform_by_pose(poses[t])
        rr.log("/keypoint_triangles", rerun_mesh_rep(kp_mesh))