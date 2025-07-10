import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

hyperparams = {
    "pose_kernel": transition_kernels.GaussianVMFPoseDriftKernel(0.02, 1000.0),
    "velocity_kernel": transition_kernels.GaussianVMFVelocityDriftKernel(0.2, 1000.0),
}

inference_hyperparams = InferenceHyperparams(
    n_poses_vels=2000,
    n_scales=100,
)
