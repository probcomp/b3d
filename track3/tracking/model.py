import jax
import jax.numpy as jnp
import genjax
from .dists import gaussian_vmf_2d

def model_factory(*,
        max_T,
        width,
        height,
        pose_kernel_params
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
    
    generate_keypoint_mesh = get_generate_keypoint_mesh()
    pose_hmm = get_pose_hmm(max_T, pose_kernel_params, width, height)
    obs_generator = get_obs_model()

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
        low = jnp.array([0., 0., 0.])
        high = jnp.array([width, height, 2*jnp.pi])
        initial_pose = genjax.uniform(low, high) @ "init"
        subsequent_poses = unfold_poses(T, initial_pose) @ "step"
        return jnp.concatenate([initial_pose[None, :], subsequent_poses])

    return pose_hmm

def get_generate_keypoint_mesh():
    @genjax.static_gen_fn
    def generate_keypoint_mesh():
        return jnp.array([0., 0., 0.])
    return generate_keypoint_mesh

def get_obs_model():
    @genjax.static_gen_fn
    def obs_model(poses, keypoint_mesh):
        return None
    return obs_model