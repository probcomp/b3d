import jax.numpy as jnp

import b3d.chisight.gen3d.image_kernel as image_kernel
import b3d.chisight.gen3d.inference as inference
import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.pixel_kernels.pixel_rgbd_kernels import (
    OldOcclusionPixelRGBDDistribution,
)

p_resample_color = 0.005
hyperparams = {
    "pose_kernel": transition_kernels.UniformPoseDriftKernel(max_shift=0.2),
    "color_kernel": transition_kernels.LaplaceNotTruncatedColorDriftKernel(scale=0.04),
    "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=jnp.array([0.001, 0.999])
    ),
    "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=jnp.array([0.001, 0.999])
    ),
    "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1,
        support=jnp.array([0.01, 0.01, 0.02]),
    ),
    "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=jnp.array([0.005, 0.1, 0.15])
    ),
    "image_kernel": image_kernel.NoOcclusionPerVertexImageKernel(
        OldOcclusionPixelRGBDDistribution()
    ),
}

inference_hyperparams = inference.InferenceHyperparams(
    n_poses=6000,
    pose_proposal_std=0.04,
    pose_proposal_conc=1000.0,
    do_stochastic_color_proposals=False,
    prev_color_proposal_laplace_scale=0.1,
    obs_color_proposal_laplace_scale=0.1,
)
