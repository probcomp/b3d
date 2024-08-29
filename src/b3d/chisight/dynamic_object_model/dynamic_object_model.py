import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

import b3d
from b3d import Pose

### Kernel from pointcloud -> image ###

# def raycast_to_image_nondeterministic(key, intrinsics, vertices_in_camera_frame, K):
#     """
#     Returns an array of shape (H, W, K) containing K point indices, or -1 to indicate no point was registered.
#     """
#     N_pts = vertices_in_camera_frame.shape[0]

#     projected_pixel_coordinates = jnp.rint(
#         b3d.xyz_to_pixel_coordinates(
#             vertices_in_camera_frame, intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
#         )
#     ).astype(jnp.int32)
#     shuffled_pixel_coordinates = jax.random.shuffle(key, projected_pixel_coordinates)

#     random_indices = jax.random.randint(key, (N_pts,), 0, K) # (N_pts,) array of random indices from 0 to K-1
#     registered_pixel_indices = -jnp.ones((intrinsics["height"], intrinsics["width"], K), dtype=int)
#     registered_pixel_indices = registered_pixel_indices.at[shuffled_pixel_coordinates, random_indices].set(jnp.arange(N_pts))

#     return registered_pixel_indices

# # def raycast_to_image_deterministic(key, args):
# #     return None
# # TODO: if the nondeterministic one seems to be making debugging hard, implement
# # a deterministic version using jnp.unique.


# # def accumulate_pixel_color(carry, projected_info):
# #     carry_color, count = carry
# #     vertex_pixel_xy, vertex_color = projected_info
# #     carry_color = carry_color.at[vertex_pixel_xy].add(vertex_color)
# #     count = count.at[vertex_pixel_xy].add(1)
# #     return (carry_color, count), None

# # (new_pixel_color, count), _ = jax.lax.scan(
# #     accumulate_pixel_color, (new_pixel_color, count), (vertex_pixel_xys, vertex_rgbs)
# # )
# # new_pixel_color = jnp.where(count > 0, new_pixel_color / count, new_pixel_color)

# #
# #
# # `pts` `colors`
# # `pts_ij` is all the points that project to pixel (i, j)
# # mean_color = mean of all point colors (ignoring "off" points)
# # mean_outlier_prob_{color/depth} = mean of all outlier probs (ignoring "off" points)

# # sampled from laplace centered at mean with color_variance, depth_variance

# # w.p. prod_{pt} nonregistration_prob[pt], no color is registered.
# # else, sample a point w.p. prop to (1 - nonregistration_prob[pt])
# # sample a color around this point from a laplace

# # Version 1 [exactly do this]
# # 1. Fill a `pts` array with K points that hit pixel (i, j).  Some slots may be empty.
# # 2.

# # Version 2 [approximate this with a mean]:
# # `pts` is the set of all points projecting to one pixel (i, j)
# # `nonregistration_prob[pt]` = probability a point is not registered, if it is the only one observed (nonregistration_prob = "outlier prob")
# # `color[pt]` = RGB or D value for the point
# # 1. Compute overall_p_nonregistered = prod_{pt} nonregistration_prob[pt].  [Set to 1.0 if pts is empty.]
# # 2. Compute the mean of all the colors for each point, where the color for `pt` is weighted proportionally to (1 - nonregistration_prob[pt])
# # 3. Sample from a mixture of [1] a uniform, with probability `nonregistration_prob[pt]`, and [2] a laplace around the mean color

# # For both of these, we should generate samples and look at them!

# @Pytree.dataclass
# class PixelDistribution(genjax.ExactDensity):
#     def sample(self, key, registered_point_indices, args):
#         """
#         Args:
#             key
#             registered_point_indices: (K,) array
#         """
#         valid_mask = (registered_point_indices >= 0)

#         sampled_inlier_colors = jax.scipy.stats.laplace.sample(
#             key, args["colors"][registered_point_indices], args["color_variance"]
#         )
#         outlier_colors = jax.random.uniform(key, args["colors"].shape)
#         corrupted = (
#             jax.random.uniform(key, len()) <
#             args["color_outlier_probability"][registered_point_indices]
#         )
#         sampled_colors = corrupted * outlier_colors + ~corrupted * sampled_inlier_colors
#         index = jax.random.categorical(jnp.log(valid_mask))


#         sampled_inlier_depth  = jax.scipy.stats.laplace.sample(
#             key, args["colors"][registered_point_indices], args["color_variance"]
#         )
#         outlier_colors = jax.random.uniform(key, args["colors"].shape)
#         corrupted = (
#             jax.random.uniform(key, len()) <
#             args["color_outlier_probability"][registered_point_indices]
#         )
#         sampled_colors = corrupted * outlier_colors + ~corrupted * sampled_inlier_colors
#         index = jax.random.categorical(jnp.log(valid_mask))


#         return sampled_colors[index]

#     def logpdf(self, observed_rgbd, likelihood_args):


# mapped_pixel_distribution = ArgMap(ImageDistFromPixelDist(
#         PixelDistribution(),
#         (True, False), # map over `registered_point_indices` per pixel, but not `args`
#     ),
#     lambda registered_point_indices, args: (H, W, *args)
# )


# @Pytree.dataclass
# class ImageDistribution(genjax.Distribution):
#     def random_weighted(key, args):
#         raycasted_image = raycast_to_image.simulate(key, args).get_retval()
#         value = mapped_pixel_distribution.sample(key, raycasted_image, args)
#         pdf_estimate = mapped_pixel_distribution.logpdf(value, raycasted_image, args)
#         return value, pdf_estimate

#     def estimate_logpdf(key, obs, args):
#         raycasted_image = raycast_to_image.simulate(key, args).get_retval()
#         pdf_estimate = mapped_pixel_distribution.logpdf(obs, raycasted_image, args)
#         return pdf_estimate


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

    colors = (
        b3d.modeling_utils.uniform_broadcasted(
            previous_state["colors"] - max_color_shift,
            previous_state["colors"] + max_color_shift,
        )
        @ "colors"
    )

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
