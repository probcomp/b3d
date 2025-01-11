import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

hyperparams = {
    "pose_kernel": transition_kernels.GaussianVMFPoseDriftKernel(0.02, 1000.0),
    "color_noise_variance": 1.0,
    "depth_noise_variance": 0.01,
    "outlier_probability": 0.1,
}

inference_hyperparams = InferenceHyperparams(
    n_poses=2000,
)
