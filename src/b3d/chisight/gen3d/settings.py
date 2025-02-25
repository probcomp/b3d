import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams

hyperparams = {
    "pose_kernel": transition_kernels.GaussianVMFPoseDriftKernel(0.02, 1000.0),
    "mu": 0.25,
    "restitution": 0.4,
    "fps": 100,
    "sim_substeps": 10,
    "g": -9.80665,
}

inference_hyperparams = InferenceHyperparams(
    n_poses=2000,
)
