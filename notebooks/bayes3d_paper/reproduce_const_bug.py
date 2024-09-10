import b3d
import b3d.chisight.gen3d.model
import b3d.chisight.gen3d.transition_kernels as transition_kernels
import b3d.io.data_loader
import jax
import jax.numpy as jnp
import numpy as np
from b3d import Pose
from genjax import ChoiceMapBuilder as C

num_vertices = 100
vertices = jax.random.uniform(
    jax.random.PRNGKey(0), (num_vertices, 3), minval=-1, maxval=1
)
colors = jax.random.uniform(
    jax.random.PRNGKey(1), (num_vertices, 3), minval=0, maxval=1
)
key = jax.random.PRNGKey(0)
hyperparams = {
    "pose_kernel": transition_kernels.UniformPoseDriftKernel(max_shift=0.1),
    "color_kernel": transition_kernels.LaplaceColorDriftKernel(scale=0.05),
    "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.01, 0.99])
    ),
    "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.01, 0.99])
    ),
    "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.005, 0.01, 0.02])
    ),
    "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.05, 0.1, 0.15])
    ),
    "vertices": vertices,
}

previous_state = {
    "pose": Pose.identity(),
    "colors": colors,
    "visibility_prob": jnp.ones(num_vertices)
    * hyperparams["visibility_prob_kernel"].support[-1],
    "depth_nonreturn_prob": jnp.ones(num_vertices)
    * hyperparams["depth_nonreturn_prob_kernel"].support[0],
    "depth_scale": hyperparams["depth_scale_kernel"].support[0],
    "color_scale": hyperparams["color_scale_kernel"].support[0],
}

key = jax.random.PRNGKey(0)

importance = b3d.chisight.gen3d.model.dynamic_object_generative_model.importance

# This line is fine.
trace = importance(key, C.n(), (hyperparams, previous_state))[0]
# This line is fine.
trace = importance(key, C.n(), (hyperparams, previous_state))[0]


# But when we reinitialize the hyperparams, we get an error.
hyperparams = {
    "pose_kernel": transition_kernels.UniformPoseDriftKernel(max_shift=0.1),
    "color_kernel": transition_kernels.LaplaceColorDriftKernel(scale=0.05),
    "visibility_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.01, 0.99])
    ),
    "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.01, 0.99])
    ),
    "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.005, 0.01, 0.02])
    ),
    "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
        resample_probability=0.1, support=np.array([0.05, 0.1, 0.15])
    ),
    "vertices": vertices,
}
trace = importance(key, C.n(), (hyperparams, previous_state))[0]
