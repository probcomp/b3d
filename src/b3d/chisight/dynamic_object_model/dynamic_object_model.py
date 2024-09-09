import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import ChoiceMapBuilder as C
from genjax import Pytree

import b3d
from b3d.chisight.dynamic_object_model.likelihoods.project_no_occlusions_kernel import (
    likelihood_func,
    sample_func,
)


@Pytree.dataclass
class ImageLikelihood(genjax.ExactDensity):
    def sample(self, key, likelihood_args):
        return sample_func(key, likelihood_args)

    def logpdf(self, observed_rgbd, likelihood_args):
        return likelihood_func(observed_rgbd, likelihood_args)["score"]


image_likelihood = ImageLikelihood()


@jax.jit
def info_from_trace(trace):
    return likelihood_func(
        trace.get_choices()["rgbd"],
        trace.get_retval()["likelihood_args"],
    )


@genjax.gen
def dynamic_object_generative_model(hyperparams, previous_state):
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

    if "image_likelihood" in hyperparams:
        likelihood_args = {
            "fx": hyperparams["fx"],
            "fy": hyperparams["fy"],
            "cx": hyperparams["cx"],
            "cy": hyperparams["cy"],
            "image_width": hyperparams["image_width"],
            "image_height": hyperparams["image_height"],
            "vertices": hyperparams["vertices"],
            "colors": colors,
            "pose": pose,
            "color_scale": color_scale,
            "depth_scale": depth_scale,
            "visibility": visibility,
            "depth_nonreturn": depth_nonreturn,
        }
        rgbd = hyperparams["image_likelihood"](likelihood_args) @ "rgbd"
    else:
        rgbd = None
        likelihood_args = None

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
        "rgbd": rgbd,
        "likelihood_args": likelihood_args,
    }


def make_colors_choicemap(colors):
    return jax.vmap(lambda idx: C["colors", idx].set(colors[idx]))(
        jnp.arange(len(colors))
    )


def make_visibiliy_choicemap(visibility):
    return jax.vmap(lambda idx: C["visibility", idx].set(visibility[idx]))(
        jnp.arange(len(visibility))
    )


def make_depth_nonreturn_choicemap(depth_nonreturn):
    return jax.vmap(lambda idx: C["depth_nonreturn", idx].set(depth_nonreturn[idx]))(
        jnp.arange(len(depth_nonreturn))
    )


### Viz ###
def viz_trace(trace, t=0, ground_truth_vertices=None, ground_truth_pose=None):
    b3d.rr_set_time(t)
    hyperparams, _ = trace.get_args()
    new_state = trace.get_retval()["new_state"]

    pose = new_state["pose"]
    colors = new_state["colors"]
    visibility = new_state["visibility"]
    depth_nonreturn = new_state["depth_nonreturn"]

    vertices = hyperparams["vertices"]
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

    vertices_transformed = pose.apply(vertices)
    b3d.rr_log_cloud(
        vertices_transformed,
        "scene/model",
        colors,
    )

    output = trace.get_retval()
    if output["rgbd"] is not None:
        info = info_from_trace(trace)
        b3d.rr_log_rgb(output["rgbd"][..., :3], "image")
        b3d.rr_log_rgb(output["rgbd"][..., :3], "image/rgb/observed")
        b3d.rr_log_depth(output["rgbd"][..., 3], "image/depth/observed")

        latent_rgbd = info["latent_rgbd"]
        b3d.rr_log_rgb(latent_rgbd[..., :3], "image/rgb/latent")
        b3d.rr_log_depth(latent_rgbd[..., 3], "image/depth/latent")

        likelihood_args = trace.get_retval()["likelihood_args"]
        fx, fy, cx, cy = (
            likelihood_args["fx"],
            likelihood_args["fy"],
            likelihood_args["cx"],
            likelihood_args["cy"],
        )
        b3d.rr_log_cloud(
            b3d.xyz_from_depth(
                output["rgbd"][..., 3],
                fx,
                fy,
                cx,
                cy,
            ),
            "scene/observed",
            output["rgbd"][..., :3].reshape(-1, 3),
        )

    if ground_truth_vertices is not None:
        b3d.rr_log_cloud(
            trace.get_choices()["pose"].apply(ground_truth_vertices),
            "scene/full_object_model",
        )

        if ground_truth_pose:
            b3d.rr_log_cloud(
                ground_truth_pose.apply(ground_truth_vertices),
                "scene/ground_truth_object_mesh",
            )
