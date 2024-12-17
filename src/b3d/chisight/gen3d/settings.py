import jax.numpy as jnp

import b3d.chisight.gen3d.image_kernel as image_kernel
import b3d.chisight.gen3d.transition_kernels as transition_kernels
from b3d.chisight.gen3d.hyperparams import InferenceHyperparams
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

p_resample_color = 0.005
hyperparams = {
    "pose_kernel": transition_kernels.GaussianVMFPoseDriftKernel(0.02, 1000.0),
    "color_noise_variance": 1,
    "depth_noise_variance": 0.01,
    "outlier_probability": 0.1,
}

inference_hyperparams = InferenceHyperparams(
    n_poses=4000,
    do_stochastic_color_proposals=False,
    prev_color_proposal_laplace_scale=0.1,
    obs_color_proposal_laplace_scale=0.1,
    # If you don't use the UniquePixelsImageKernel, you should probably
    # change this to False.
    in_inference_only_assoc_one_point_per_pixel=True,
)


# WIP_hyperparams = {
#     "pose_kernel": transition_kernels.GaussianVMFPoseDriftKernel(0.02, 1000.0),
#     "color_kernel": transition_kernels.RenormalizedLaplaceColorDriftKernel(scale=0.002),
#     # transition_kernels.MixtureDriftKernel(
#     #     [
#     #         transition_kernels.RenormalizedLaplaceColorDriftKernel(scale=0.01),
#     #         transition_kernels.UniformDriftKernel(
#     #             max_shift=0.15, min_val=jnp.zeros(3), max_val=jnp.ones(3)
#     #         ),
#     #     ],
#     #     jnp.array([0.8, 1 - 0.805, .005]),
#     # ),
#     "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
#         p_change_to_different_value=0.05, support=jnp.array([0.0, 1.0])
#     ),
#     "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
#         p_change_to_different_value=0.1, support=jnp.array([0.0, 1.0])
#     ),
#     "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
#         p_change_to_different_value=0.1,
#         support=jnp.array([0.001, 0.0025, 0.01, 0.02]),
#     ),
#     "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
#         p_change_to_different_value=0.1,
#         support=jnp.array([0.002, 0.01, 0.02, 0.05, 0.1, 0.15]),
#     ),
#     "image_kernel": image_kernel.UniquePixelsImageKernel(
#         FullPixelRGBDDistribution(
#             RenormalizedLaplacePixelColorDistribution(),
#             UniformPixelColorDistribution(),
#             RenormalizedLaplacePixelDepthDistribution(),
#             UniformPixelDepthDistribution(),
#         )
#     ),
#     "unexplained_depth_nonreturn_prob": 0.02,
# }
