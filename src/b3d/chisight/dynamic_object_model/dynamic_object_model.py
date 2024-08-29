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


@jax.jit
def info_from_trace(trace):
    return laplace_no_rendering_likelihood_function(
        trace.get_choices()["rgbd"],
        trace.get_retval()["args"],
    )


### Step model ###


@genjax.gen
def dynamic_object_generative_model(args):
    num_vertices = args["num_vertices"].const

    vertices = (
        b3d.modeling_utils.uniform_broadcasted(
            -1.0 * jnp.ones((num_vertices, 3)), 1.0 * jnp.ones((num_vertices, 3))
        )
        @ "vertices"
    )
    args["vertices"] = vertices

    old_pose = (
        Pose.uniform_pose_centered(
            Pose.identity(), -100.0 * jnp.ones(3), 100.0 * jnp.ones(3)
        )
        @ "old_pose"
    )

    pose = (
        Pose.uniform_pose_centered(old_pose, -0.1 * jnp.ones(3), 0.1 * jnp.ones(3))
        @ "pose"
    )
    args["pose"] = pose

    old_colors = (
        b3d.modeling_utils.uniform_broadcasted(
            jnp.zeros((num_vertices, 3)), jnp.ones((num_vertices, 3))
        )
        @ "old_colors"
    )

    max_color_shift = args["max_color_shift"].const
    colors = (
        b3d.modeling_utils.uniform_broadcasted(
            old_colors - max_color_shift, old_colors + max_color_shift
        )
        @ "colors"
    )
    args["colors"] = colors

    color_outlier_probability = (
        b3d.modeling_utils.uniform_broadcasted(
            0.0 * jnp.ones((num_vertices,)), 1.0 * jnp.ones((num_vertices,))
        )
        @ "color_outlier_probability"
    )
    args["color_outlier_probability"] = color_outlier_probability

    depth_outlier_probability = (
        b3d.modeling_utils.uniform_broadcasted(
            0.0 * jnp.ones((num_vertices,)), 1.0 * jnp.ones((num_vertices,))
        )
        @ "depth_outlier_probability"
    )
    args["depth_outlier_probability"] = depth_outlier_probability

    depth_variance = genjax.uniform(0.0001, 100000.0) @ "depth_variance"
    color_variance = genjax.uniform(0.0001, 100000.0) @ "color_variance"
    args["color_variance"] = color_variance
    args["depth_variance"] = depth_variance

    rgbd = image_likelihood(args) @ "rgbd"
    return {"args": args, "rgbd": rgbd}


### Viz ###


def viz_trace(trace, t=0, ground_truth_vertices=None, ground_truth_pose=None):
    info = info_from_trace(trace)
    b3d.utils.rr_set_time(t)
    args = trace.get_retval()["args"]
    fx, fy, cx, cy = (
        args["fx"],
        args["fy"],
        args["cx"],
        args["cy"],
    )

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
        trace.get_choices()["vertices"],
        "object/model",
        trace.get_choices()["colors"],
    )
    b3d.rr_log_cloud(
        trace.get_choices()["vertices"],
        "object/color_outlier_probability",
        jnp.array([[1.0, 0.0, 0.0]])
        * trace.get_choices()["color_outlier_probability"][:, None],
    )
    b3d.rr_log_cloud(
        trace.get_choices()["vertices"],
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
