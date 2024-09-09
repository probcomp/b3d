### IMPORTS ###

import b3d
import b3d.chisight.gen3d.model
import b3d.chisight.gen3d.transition_kernels as transition_kernels
import jax
import jax.numpy as jnp
from b3d import Pose
from genjax import ChoiceMapBuilder as C

b3d.rr_init("test_dynamic_object_model")


def test_model_no_likelihood():
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
            resample_probability=0.1, possible_values=jnp.array([0.01, 0.99])
        ),
        "depth_nonreturn_prob_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, possible_values=jnp.array([0.01, 0.99])
        ),
        "depth_scale_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, possible_values=jnp.array([0.005, 0.01, 0.02])
        ),
        "color_scale_kernel": transition_kernels.DiscreteFlipKernel(
            resample_probability=0.1, possible_values=jnp.array([0.05, 0.1, 0.15])
        ),
        "vertices": vertices,
    }

    previous_state = {
        "pose": Pose.identity(),
        "colors": colors,
        "visibility_prob": jnp.ones(num_vertices)
        * hyperparams["visibility_prob_kernel"].possible_values[-1],
        "depth_nonreturn_prob": jnp.ones(num_vertices)
        * hyperparams["depth_nonreturn_prob_kernel"].possible_values[0],
        "depth_scale": hyperparams["depth_scale_kernel"].possible_values[0],
        "color_scale": hyperparams["color_scale_kernel"].possible_values[0],
    }

    key = jax.random.PRNGKey(0)
    importance = jax.jit(
        b3d.chisight.gen3d.model.dynamic_object_generative_model.importance
    )

    trace, _ = importance(key, C.n(), (hyperparams, previous_state))
    assert trace.get_score().shape == ()

    traces = [trace]
    for t in range(100):
        key = b3d.split_key(key)
        previous_state = trace.get_retval()["new_state"]
        trace, _ = importance(key, C.n(), (hyperparams, previous_state))
        b3d.chisight.gen3d.model.viz_trace(trace, t)
        traces.append(trace)
    colors_over_time = jnp.array(
        [trace.get_choices()["colors", ...] for trace in traces]
    )

    from IPython import embed

    embed()
    # Subplots

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 15))
    point_index = 0
    fig.suptitle(f"Properities of vertex {point_index} over time")
    ax[0].set_title(f"Color of vertex {point_index}")
    ax[0].plot(colors_over_time[..., point_index, 0], color="r")
    ax[0].plot(colors_over_time[..., point_index, 1], color="g")
    ax[0].plot(colors_over_time[..., point_index, 2], color="b")
    ax[0].set_ylim(-0.01, 1.01)

    first_n = 10
    ax[1].set_title("Visibility")
    ax[1].plot(
        [trace.get_choices()["visibility_prob", ...][:first_n] for trace in traces],
        alpha=0.5,
    )

    ax[2].set_title("Depth Non Return")
    ax[2].plot(
        [
            trace.get_choices()["depth_nonreturn_prob", ...][:first_n]
            for trace in traces
        ],
        alpha=0.5,
    )

    ax[3].set_title("Inlier Scale")
    ax[3].plot([trace.get_choices()["depth_scale"] for trace in traces], label="depth")
    ax[3].plot([trace.get_choices()["color_scale"] for trace in traces], label="color")
    ax[3].legend()
    fig.supxlabel("Time")
    fig.savefig("test_dynamic_object_model.png")


if __name__ == "__main__":
    test_model_no_likelihood()
