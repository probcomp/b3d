import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import ChoiceMapBuilder as C
from genjax import Pytree

import b3d
from b3d import Pose

LIKELIHOOD = "aggregate_mean"
LIKELIHOOD = "project_no_occlusions"

if LIKELIHOOD == "project_no_occlusions":
    from b3d.chisight.dynamic_object_model.likelihoods.project_no_occlusions_kernel import (
        likelihood_func,
        sample_func,
    )
elif LIKELIHOOD == "aggregate_mean":
    from b3d.chisight.dynamic_object_model.likelihoods.aggreate_mean_image_kernel import (
        likelihood_func,
        sample_func,
    )
else:
    raise NotImplementedError(f"Unknown likelihood: {LIKELIHOOD}")


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


@Pytree.dataclass
class ColorTransitionKernel(genjax.ExactDensity):
    def sample(self, key, color, scale):
        return genjax.laplace.sample(key, color, scale)

    def logpdf(self, new_color, color, scale):
        return jax.scipy.stats.laplace.logpdf(new_color, color, scale).sum()


color_transition_kernel = ColorTransitionKernel()
vectorized_color_transition_kernel_logpdf = jnp.vectorize(
    color_transition_kernel.logpdf, signature="(3),(3),()->()"
)


@Pytree.dataclass
class OutlierProbabilityTransitionKernel(genjax.ExactDensity):
    def sample(self, key, outlier_probability, scale):
        return genjax.laplace.sample(key, outlier_probability, scale)

    def logpdf(self, new_outlier_probability, outlier_probability, scale):
        return jax.scipy.stats.laplace.logpdf(
            new_outlier_probability, outlier_probability, scale
        ).sum()


outlier_probability_transition_kernel = OutlierProbabilityTransitionKernel()
vectorized_outlier_probability_transition_kernel_logpdf = jnp.vectorize(
    outlier_probability_transition_kernel.logpdf, signature="(),(),()->()"
)


@genjax.gen
def dynamic_object_generative_model(hyperparams, previous_state):
    vertices = hyperparams["vertices"]

    pose = (
        Pose.uniform_pose_centered(
            previous_state["pose"],
            -hyperparams["max_pose_position_shift"] * jnp.ones(3),
            hyperparams["max_pose_position_shift"] * jnp.ones(3),
        )
        @ "pose"
    )

    # TODO: change this to a mixture of 0.02 * this distribution + 0.98 * a very tight laplace around the old colors
    colors = (
        color_transition_kernel.vmap(in_axes=(0, None))(
            previous_state["colors"], hyperparams["color_shift_scale"]
        )
        @ "colors"
    )

    color_outlier_probabilities = (
        outlier_probability_transition_kernel.vmap(in_axes=(0, None))(
            previous_state["color_outlier_probabilities"],
            hyperparams["color_outlier_probability_shift_scale"],
        )
        @ "color_outlier_probabilities"
    )
    depth_outlier_probabilities = (
        outlier_probability_transition_kernel.vmap(in_axes=(0, None))(
            previous_state["depth_outlier_probabilities"],
            hyperparams["depth_outlier_probability_shift_scale"],
        )
        @ "depth_outlier_probabilities"
    )

    depth_variance = genjax.uniform(0.0001, 100000.0) @ "depth_variance"
    color_variance = genjax.uniform(0.0001, 100000.0) @ "color_variance"

    likelihood_args = {
        "fx": hyperparams["fx"],
        "fy": hyperparams["fy"],
        "cx": hyperparams["cx"],
        "cy": hyperparams["cy"],
        "image_height": hyperparams["image_height"],
        "image_width": hyperparams["image_width"],
        "vertices": vertices,
        "pose": pose,
        "colors": colors,
        "color_outlier_probabilities": color_outlier_probabilities,
        "depth_outlier_probabilities": depth_outlier_probabilities,
        "depth_variance": depth_variance,
        "color_variance": color_variance,
    }
    rgbd = image_likelihood(likelihood_args) @ "rgbd"

    new_state = {
        "pose": pose,
        "colors": colors,
        "color_outlier_probabilities": color_outlier_probabilities,
        "depth_outlier_probabilities": depth_outlier_probabilities,
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


def make_color_outlier_probabilities_choicemap(color_outlier_probabilities):
    return jax.vmap(
        lambda idx: C["color_outlier_probabilities", idx].set(
            color_outlier_probabilities[idx]
        )
    )(jnp.arange(len(color_outlier_probabilities)))


def make_depth_outlier_probabilities_choicemap(depth_outlier_probabilities):
    return jax.vmap(
        lambda idx: C["depth_outlier_probabilities", idx].set(
            depth_outlier_probabilities[idx]
        )
    )(jnp.arange(len(depth_outlier_probabilities)))


### Viz ###
def viz_trace(trace, t=0, ground_truth_vertices=None, ground_truth_pose=None):
    info = info_from_trace(trace)
    b3d.utils.rr_set_time(t)
    likelihood_args = trace.get_retval()["likelihood_args"]
    fx, fy, cx, cy = (
        likelihood_args["fx"],
        likelihood_args["fy"],
        likelihood_args["cx"],
        likelihood_args["cy"],
    )
    vertices = trace.get_args()[0]["vertices"]

    info = info_from_trace(trace)
    rr.log("image", rr.Image(trace.get_choices()["rgbd"][..., :3]))
    b3d.rr_log_rgb(trace.get_choices()["rgbd"][..., :3], "image/rgb/observed")
    b3d.rr_log_rgb(info["latent_rgbd"][..., :3], "image/rgb/latent")
    b3d.rr_log_depth(trace.get_choices()["rgbd"][..., 3], "image/depth/observed")
    b3d.rr_log_depth(info["latent_rgbd"][..., 3], "image/depth/latent")

    b3d.rr_log_cloud(
        info["transformed_points"],
        "scene/latent",
        trace.get_choices()["colors", ...],
    )
    b3d.rr_log_cloud(
        b3d.xyz_from_depth(
            trace.get_retval()["rgbd"][..., 3],
            fx,
            fy,
            cx,
            cy,
        ),
        "scene/observed",
        trace.get_retval()["rgbd"][..., :3].reshape(-1, 3),
    )

    b3d.rr_log_cloud(
        vertices,
        "object/model",
        trace.get_choices()["colors", ...],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/color_outlier_probabilities",
        jnp.array([[1.0, 0.0, 0.0]])
        * trace.get_choices()["color_outlier_probabilities", ...][:, None],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/depth_outlier_probabilities",
        jnp.array([[0.0, 1.0, 0.0]])
        * trace.get_choices()["depth_outlier_probabilities", ...][:, None],
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
