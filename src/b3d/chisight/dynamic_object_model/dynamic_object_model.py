import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U

import b3d


@genjax.gen
def dynamic_object_generative_model_no_likelihood(hyperparams, previous_state):
    hyperparams["vertices"]
    pose_transition_kernel = hyperparams["pose_transition_kernel"]
    color_transition_kernel = hyperparams["color_transition_kernel"]

    visiblity_values = hyperparams["visibility_values"]
    visibility_transition_kernel = hyperparams["visibility_transition_kernel"]

    depth_nonreturn_values = hyperparams["depth_nonreturn_values"]
    depth_nonreturn_transition_kernel = hyperparams["depth_nonreturn_transition_kernel"]

    depth_scale_values = hyperparams["depth_scale_values"]
    depth_scale_transition_kernel = hyperparams["depth_scale_transition_kernel"]

    color_scale_values = hyperparams["color_scale_values"]
    color_scale_transition_kernel = hyperparams["color_scale_transition_kernel"]

    pose = pose_transition_kernel(previous_state["pose"]) @ "pose"
    colors = color_transition_kernel.vmap()(previous_state["colors"]) @ "colors"

    visibility = (
        visibility_transition_kernel.vmap(in_axes=(0, None))(
            previous_state["visibility"], visiblity_values
        )
        @ "visibility"
    )
    depth_nonreturn = (
        depth_nonreturn_transition_kernel.vmap(in_axes=(0, None))(
            previous_state["depth_nonreturn"], depth_nonreturn_values
        )
        @ "depth_nonreturn"
    )
    depth_scale = (
        depth_scale_transition_kernel(previous_state["depth_scale"], depth_scale_values)
        @ "depth_scale"
    )
    color_scale = (
        color_scale_transition_kernel(previous_state["color_scale"], color_scale_values)
        @ "color_scale"
    )

    new_state = {
        "pose": pose,
        "colors": colors,
        "visibility": visibility,
        "depth_nonreturn": depth_nonreturn,
        "depth_scale": depth_scale,
        "color_scale": color_scale,
    }
    return {
        "new_state": new_state,
    }


### Viz ###
def viz_trace(trace, t=0):
    b3d.rr_set_time(t)
    hyperparams, _ = trace.get_args()
    new_state = trace.get_retval()["new_state"]

    # pose = new_state["pose"]
    colors = new_state["colors"]
    visibility = new_state["visibility"]
    depth_nonreturn = new_state["depth_nonreturn"]

    vertices = hyperparams["vertices"]
    # vertices_transformed = pose.apply(vertices)
    b3d.rr_log_cloud(
        vertices,
        "object/model",
        colors,
    )
    b3d.rr_log_cloud(
        vertices,
        "object/visibility",
        jnp.array([[1.0, 0.0, 0.0]]) * visibility[..., None],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/depth_nonreturn",
        jnp.array([[0.0, 1.0, 0.0]]) * depth_nonreturn[..., None],
    )

    rr.log(
        "info",
        rr.TextDocument(
            f"""
            depth_scale: {new_state["depth_scale"]}
            color_scale: {new_state["color_scale"]}
            """.strip(),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )


@jax.jit
def advance_time(key, trace, observed_rgbd):
    """
    Advance to the next timestep, setting the new latent state to the
    same thing as the previous latent state, and setting the new
    observed RGBD value.

    Returns a trace where previous_state (stored in the arguments)
    and new_state (sampled in the choices and returned) are identical.
    """
    hyperparams, _ = trace.get_args()
    previous_state = trace.get_retval()["new_state"]
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (Diff.no_change(hyperparams), Diff.unknown_change(previous_state)),
            C.kw(rgbd=observed_rgbd),
        ),
    )
    return trace
