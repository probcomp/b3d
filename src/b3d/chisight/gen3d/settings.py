import jax.numpy as jnp

import b3d.chisight.gen3d.image_kernel as image_kernel
import b3d.chisight.gen3d.inference as inference
import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.pixel_kernels.pixel_color_kernels import (
    RenormalizedLaplacePixelColorDistribution,
    UniformPixelColorDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_depth_kernels import (
    RenormalizedLaplacePixelDepthDistribution,
    UniformPixelDepthDistribution,
)
from b3d.chisight.gen3d.pixel_kernels.pixel_rgbd_kernels import (
    FullPixelRGBDDistribution,
)

p_resample_color = 0.03
hyperparams = {
    "pose_kernel": {
        "distribution": transition_kernels.GaussianVMFPoseDriftKernel(),
        "args": (0.02, 1000.0),
    },
    "color_kernel": transition_kernels.MixtureDriftKernel(
        [
            transition_kernels.LaplaceNotTruncatedColorDriftKernel(scale=0.03),
            transition_kernels.UniformDriftKernel(
                max_shift=0.15, min_val=jnp.zeros(3), max_val=jnp.ones(3)
            ),
        ],
        jnp.array([1 - p_resample_color, p_resample_color]),
    ),
    "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=jnp.array([0.001, 0.999])
    ),
    "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.3, support=jnp.array([0.001, 0.999])
    ),
    "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1,
        support=jnp.array([0.0025, 0.01, 0.02]),
    ),
    "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=jnp.array([0.002, 0.01, 0.05, 0.1, 0.15, 0.3])
    ),
    "image_kernel": image_kernel.UniquePixelsImageKernel(
        FullPixelRGBDDistribution(
            RenormalizedLaplacePixelColorDistribution(),
            UniformPixelColorDistribution(),
            RenormalizedLaplacePixelDepthDistribution(),
            UniformPixelDepthDistribution(),
        )
    ),
    "unexplained_depth_nonreturn_prob": 0.02,
}

inference_hyperparams = inference.InferenceHyperparams(
    n_poses=4000,
    pose_proposal_std=0.04,
    pose_proposal_conc=1000.0,
    do_stochastic_color_proposals=False,
    prev_color_proposal_laplace_scale=0.04,
    obs_color_proposal_laplace_scale=0.01,
)
