### IMPORTS ###
import os

import b3d
import b3d.chisight.gen3d.model
import b3d.chisight.gen3d.settings
import b3d.io.data_loader
import jax
import jax.numpy as jnp
from b3d import Mesh, Pose
from b3d.chisight.gen3d.model import (
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
)
from genjax import ChoiceMapBuilder as C
from genjax import Pytree

b3d.rr_init("test_gen3d_model")


def test_model():
    importance = b3d.chisight.gen3d.model.dynamic_object_generative_model.importance

    # num_vertices = 100
    # vertices = jax.random.uniform(
    #     jax.random.PRNGKey(0), (num_vertices, 3), minval=-1, maxval=1
    # )
    # colors = jax.random.uniform(
    #     jax.random.PRNGKey(1), (num_vertices, 3), minval=0, maxval=1
    # )

    ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")
    id = 0
    mesh = Mesh.from_obj_file(
        os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')
    ).scale(0.001)
    vertices = mesh.vertices
    colors = mesh.vertex_attributes
    num_vertices = vertices.shape[0]

    key = jax.random.PRNGKey(0)

    hyperparams = b3d.chisight.gen3d.settings.hyperparams

    hyperparams["vertices"] = vertices
    hyperparams["intrinsics"] = {
        "image_height": Pytree.const(480),
        "image_width": Pytree.const(640),
        "fx": 1066.778,
        "fy": 1067.487,
        "cx": 312.9869,
        "cy": 241.3109,
        "near": 0.1,
        "far": 10.0,
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
    trace = importance(key, C.n(), (hyperparams, previous_state))[0]
    trace = importance(key, C.n(), (hyperparams, previous_state))[0]

    key = jax.random.PRNGKey(0)
    hyperparams, previous_state = trace.get_args()

    traces = [trace]
    for t in range(10):
        key = b3d.split_key(key)
        previous_state = trace.get_retval()["new_state"]
        trace, _ = importance(key, C.n(), (hyperparams, previous_state))
        b3d.chisight.gen3d.model.viz_trace(trace, t)
        traces.append(trace)

    colors_over_time = jnp.array(
        [trace.get_choices()["colors", ...] for trace in traces]
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 20))
    point_index = 0

    fig.suptitle(
        f"""
pose_kernel max_shift: FILL IN,
color_kernel scale: FILL IN,
visibility_prob_kernel resample_probability: {hyperparams['visibility_prob_kernel'].resample_probability},
depth_nonreturn_prob_kernel resample_probability: {hyperparams['depth_nonreturn_prob_kernel'].resample_probability},
depth_scale_kernel resample_probability: {hyperparams['depth_scale_kernel'].resample_probability},
color_scale_kernel resample_probability: {hyperparams['color_scale_kernel'].resample_probability}"""
    )
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
    fig.savefig("test_gen3d_model.png")

    colors = trace.get_choices()["colors", ...]
    new_colors = colors + 0.01
    new_colors_choicemap = make_colors_choicemap(new_colors)
    new_trace = trace.update(key, new_colors_choicemap)[0]
    assert jnp.allclose(new_trace.get_choices()["colors", ...], new_colors)

    visibility_prob = trace.get_choices()["visibility_prob", ...]
    new_visibility_prob = visibility_prob + 0.01
    new_visibility_prob_choicemap = make_visibility_prob_choicemap(new_visibility_prob)
    new_trace = trace.update(key, new_visibility_prob_choicemap)[0]
    assert jnp.allclose(
        new_trace.get_choices()["visibility_prob", ...], new_visibility_prob
    )

    depth_nonreturn_prob = trace.get_choices()["depth_nonreturn_prob", ...]
    new_depth_nonreturn_prob = depth_nonreturn_prob + 0.01
    new_depth_nonreturn_prob_choicemap = make_depth_nonreturn_prob_choicemap(
        new_depth_nonreturn_prob
    )
    new_trace = trace.update(key, new_depth_nonreturn_prob_choicemap)[0]
    assert jnp.allclose(
        new_trace.get_choices()["depth_nonreturn_prob", ...], new_depth_nonreturn_prob
    )
