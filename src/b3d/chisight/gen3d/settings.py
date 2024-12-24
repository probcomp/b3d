import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

hyperparams = {
    "pose_kernel": transition_kernels.PhysicsPoseKernel(0.001, 1000.0),
    "vel_kernel": transition_kernels.GaussianVelocityDriftKernel(0.06),
    # "ang_vel_kernel": transition_kernels.GaussianVelocityDriftKernel(0.01),
    "color_noise_variance": 1,
    "depth_noise_variance": 0.01,
    "outlier_probability": 0.1,
}

inference_hyperparams = InferenceHyperparams(
    n_poses=2000,
)
