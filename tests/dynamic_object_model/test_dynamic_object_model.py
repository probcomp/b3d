### IMPORTS ###


import os

import b3d
import b3d.chisight.dynamic_object_model.drift_kernels
import b3d.chisight.dynamic_object_model.drift_kernels as drift_kernels
import b3d.chisight.dynamic_object_model.dynamic_object_model
import jax
import jax.numpy as jnp
from b3d import Mesh, Pose
from genjax import ChoiceMapBuilder as C

b3d.rr_init("test_dynamic_object_model")


def test_dynamic_object_generative_model():
    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")
    id = 0
    mesh = Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)

    vertices = mesh.vertices
    colors = mesh.vertex_attributes

    key = jax.random.PRNGKey(0)

    hyperparams = {
        "pose_transition_kernel": drift_kernels.UniformPoseDriftKernel(max_shift=0.1),
        "color_transition_kernel": drift_kernels.LaplaceColorDriftKernel(scale=0.05),
        "visibility_transition_kernel": drift_kernels.DiscreteFlipKernel(
            resample_probability=0.1
        ),
        "depth_nonreturn_transition_kernel": drift_kernels.DiscreteFlipKernel(
            resample_probability=0.1
        ),
        "depth_scale_transition_kernel": drift_kernels.DiscreteFlipKernel(
            resample_probability=0.1
        ),
        "color_scale_transition_kernel": drift_kernels.DiscreteFlipKernel(
            resample_probability=0.1
        ),
        "visibility_values": jnp.array([0.01, 0.99]),
        "depth_nonreturn_values": jnp.array([0.01, 0.99]),
        "depth_scale_values": jnp.array([0.005, 0.01, 0.02]),
        "color_scale_values": jnp.array([0.05, 0.1, 0.15]),
        "vertices": vertices,
    }

    num_vertices = vertices.shape[0]
    previous_state = {
        "pose": Pose.identity(),
        "colors": colors,
        "visibility": jnp.ones(num_vertices) * hyperparams["visibility_values"][1],
        "depth_nonreturn": jnp.ones(num_vertices)
        * hyperparams["depth_nonreturn_values"][0],
        "depth_scale": hyperparams["depth_scale_values"][0],
        "color_scale": hyperparams["color_scale_values"][0],
    }

    key = jax.random.PRNGKey(0)
    importance = jax.jit(
        b3d.chisight.dynamic_object_model.dynamic_object_model.dynamic_object_generative_model_no_likelihood.importance
    )

    trace, _ = importance(key, C.n(), (hyperparams, previous_state))
    assert trace.get_score().shape == ()

    traces = [trace]
    for t in range(100):
        key = b3d.split_key(key)
        previous_state = trace.get_retval()["new_state"]
        trace, _ = importance(key, C.n(), (hyperparams, previous_state))
        b3d.chisight.dynamic_object_model.dynamic_object_model.viz_trace(trace, t)
        print(trace.get_score())
        traces.append(trace)

        colors_over_time = jnp.array(
            [trace.get_choices()["colors", ...] for trace in traces]
        )

    # Subplots
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    point_index = 0
    fig.suptitle(f"Properities of vertex {point_index} over time")
    ax[0, 0].set_title("Color")
    ax[0, 0].plot(colors_over_time[..., point_index, 0], color="r")
    ax[0, 0].plot(colors_over_time[..., point_index, 1], color="g")
    ax[0, 0].plot(colors_over_time[..., point_index, 2], color="b")
    ax[0, 0].set_ylim(-0.01, 1.01)

    ax[0, 1].set_title("Visibility")
    ax[0, 1].plot(
        [trace.get_choices()["visibility", ...][point_index] for trace in traces]
    )

    ax[1, 0].set_title("Depth Non Return")
    ax[1, 0].plot(
        [trace.get_choices()["depth_nonreturn", ...][point_index] for trace in traces]
    )

    ax[1, 1].set_title("Inlier Scale")
    ax[1, 1].plot(
        [trace.get_choices()["depth_scale"] for trace in traces], label="depth"
    )
    ax[1, 1].plot(
        [trace.get_choices()["color_scale"] for trace in traces], label="color"
    )
    ax[1, 1].legend()

    fig.savefig("test_dynamic_object_model.png")
