import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

import b3d
from b3d import Pose

### Kernel from pointcloud -> image ###


@jax.jit
def laplace_no_rendering_likelihood_function(observed_rgbd, args):
    transformed_points = args["pose"].apply(args["vertices"])

    projected_pixel_coordinates = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            transformed_points, args["fx"], args["fy"], args["cx"], args["cy"]
        )
    ).astype(jnp.int32)

    observed_rgbd_masked = observed_rgbd[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1]
    ]

    color_outlier_probability = args["color_outlier_probability"]
    depth_outlier_probability = args["depth_outlier_probability"]

    color_probability = jnp.logaddexp(
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., :3], args["colors"], args["color_variance"]
        ).sum(axis=-1)
        + jnp.log(1 - color_outlier_probability),
        jnp.log(color_outlier_probability) * jnp.log(1 / 1.0**3),  # <- log(1) == 0 tho
    )
    depth_probability = jnp.logaddexp(
        jax.scipy.stats.laplace.logpdf(
            observed_rgbd_masked[..., 3],
            transformed_points[..., 2],
            args["depth_variance"],
        )
        + jnp.log(1 - depth_outlier_probability),
        jnp.log(depth_outlier_probability) * jnp.log(1 / 1.0),
    )

    scores = color_probability + depth_probability

    # Visualization
    latent_rgbd = jnp.zeros_like(observed_rgbd)
    latent_rgbd = latent_rgbd.at[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], :3
    ].set(args["colors"])
    latent_rgbd = latent_rgbd.at[
        projected_pixel_coordinates[..., 0], projected_pixel_coordinates[..., 1], 3
    ].set(transformed_points[..., 2])

    return {
        "score": scores.sum(),
        "scores": scores,
        "transformed_points": transformed_points,
        "observed_rgbd_masked": observed_rgbd_masked,
        "color_probability": color_probability,
        "depth_probability": depth_probability,
        "latent_rgbd": latent_rgbd,
    }


def sample_func(key, likelihood_args):
    return jnp.zeros(
        (
            likelihood_args["image_height"].const,
            likelihood_args["image_width"].const,
            4,
        )
    )


@Pytree.dataclass
class ImageLikelihood(genjax.ExactDensity):
    def sample(self, key, likelihood_args):
        return sample_func(key, likelihood_args)

    def logpdf(self, observed_rgbd, likelihood_args):
        return laplace_no_rendering_likelihood_function(observed_rgbd, likelihood_args)[
            "score"
        ]


image_likelihood = ImageLikelihood()

# @genjax.gen
# def likelihood(args):
#     metadata = f()
#     image = image_likelihood(metadata)


@jax.jit
def info_from_trace(trace):
    return laplace_no_rendering_likelihood_function(
        trace.get_choices()["rgbd"],
        trace.get_retval()["likelihood_args"],
    )


### Step model ###

# 1. (theta, z0) ~ P_0
# 2. For t > 0:
#     z_t ~ P_step(. ; theta, z_{t-1})

# P_0 --> P*(hyperparams_0, dummy_z)
# P_step -> P*(hyperparams_step, z_{t-1})
# We are implementing P*.


@genjax.gen
def dynamic_object_generative_model(hyperparams, previous_state):
    max_color_shift = hyperparams["max_color_shift"].const
    max_pose_position_shift = hyperparams["max_pose_position_shift"].const
    vertices = hyperparams["vertices"]
    num_vertices = vertices.shape[0]

    pose = (
        Pose.uniform_pose_centered(
            previous_state["pose"],
            -max_pose_position_shift * jnp.ones(3),
            max_pose_position_shift * jnp.ones(3),
        )
        @ "pose"
    )

    # TODO: change this to a mixture of 0.02 * this distribution + 0.98 * a very tight laplace around the old colors
    colors = (
        b3d.modeling_utils.uniform_broadcasted(
            previous_state["colors"] - max_color_shift,
            previous_state["colors"] + max_color_shift,
        )
        @ "colors"
    )

    # TODO: gradual change on the outlier probabilities and variance values
    # One way: mixture of 1% switch to a uniformly new value, and 99% sample
    # from a tight laplace/normal around the old value

    color_outlier_probability = (
        b3d.modeling_utils.uniform_broadcasted(
            0.0 * jnp.ones((num_vertices,)), 1.0 * jnp.ones((num_vertices,))
        )
        @ "color_outlier_probability"
    )
    depth_outlier_probability = (
        b3d.modeling_utils.uniform_broadcasted(
            0.0 * jnp.ones((num_vertices,)), 1.0 * jnp.ones((num_vertices,))
        )
        @ "depth_outlier_probability"
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
        "color_outlier_probability": color_outlier_probability,
        "depth_outlier_probability": depth_outlier_probability,
        "depth_variance": depth_variance,
        "color_variance": color_variance,
    }
    rgbd = image_likelihood(likelihood_args) @ "rgbd"

    new_state = {
        "pose": pose,
        "colors": colors,
    }
    return {
        "new_state": new_state,
        "rgbd": rgbd,
        "likelihood_args": likelihood_args,
    }


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
        trace.get_choices()["colors"],
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
        trace.get_choices()["colors"],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/color_outlier_probability",
        jnp.array([[1.0, 0.0, 0.0]])
        * trace.get_choices()["color_outlier_probability"][:, None],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/depth_outlier_probability",
        jnp.array([[0.0, 1.0, 0.0]])
        * trace.get_choices()["depth_outlier_probability"][:, None],
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
