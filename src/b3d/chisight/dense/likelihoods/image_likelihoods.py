import genjax
from genjax.generative_functions.distributions import ExactDensity
import jax.numpy as jnp
import b3d

# from b3d import Mesh, Pos
from collections import namedtuple
from b3d.modeling_utils import uniform_discrete, uniform_pose
from b3d.camera import unproject_depth
import jax
import os
import functools


# @jax.jit
def gaussian_depth_likelihood(observed_depth, rendered_depth, depth_variance):
    probabilities = jax.scipy.stats.norm.logpdf(
        observed_depth, rendered_depth, depth_variance
    )
    return probabilities


# @jax.jit
def gaussian_rgb_likelihood(observed_rgb, rendered_rgb, lab_variance):
    probabilities = jax.scipy.stats.norm.logpdf(
        b3d.colors.rgb_to_lab(observed_rgb),
        b3d.colors.rgb_to_lab(rendered_rgb),
        lab_variance,
    )
    return probabilities


# @jax.jit
def gaussian_iid_pix_likelihood(observed_rgbd, likelihood_args):
    rgb_variance = likelihood_args["rgb_tolerance"]
    depth_variance = likelihood_args["depth_tolerance"]
    outlier_prob = likelihood_args["outlier_prob"]
    rendered_rgbd = likelihood_args["latent_rgbd"]

    rgb_score = gaussian_rgb_likelihood(
        observed_rgbd[..., :3], rendered_rgbd[..., :3], rgb_variance
    ).sum(axis=2)
    depth_score = gaussian_depth_likelihood(
        observed_rgbd[..., 3], rendered_rgbd[..., 3], depth_variance
    )
    probabilities = rgb_score + depth_score

    probabilities_adjusted = jnp.logaddexp(
        probabilities + jnp.log(1.0 - outlier_prob), jnp.log(outlier_prob)
    )
    return probabilities_adjusted.sum(), {
        "pix_score": probabilities_adjusted,
        "rgb_pix_score": rgb_score,
        "depth_pix_score": depth_score,
    }


# gaussian_iid_pix_likelihood_vec = jax.jit(
#     jax.vmap(gaussian_iid_pix_likelihood, in_axes=(None, 0, None))
# )


###########
@functools.partial(
    jnp.vectorize,
    signature="(m)->()",
    excluded=(
        1,
        2,
        3,
        4,
        5,
        6,
    ),
)
def gausssian_mixture_vectorize_old(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    distances = observed_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(
        rendered_xyz_padded,
        (ij[0], ij[1], 0),
        (2 * filter_size + 1, 2 * filter_size + 1, 3),
    )
    probabilities = jax.scipy.stats.norm.logpdf(
        distances, loc=0.0, scale=jnp.sqrt(variance)
    ).sum(-1) - jnp.log(observed_xyz.shape[0] * observed_xyz.shape[1])
    # return jnp.logaddexp(
    #     probabilities.max() + jnp.log(1.0 - outlier_prob),
    #     jnp.log(outlier_prob) - jnp.log(outlier_volume),
    # )
    inlier_score = probabilities.max() + jnp.log(1.0 - outlier_prob)
    outlier_score = jnp.log(outlier_prob) - jnp.log(outlier_volume)
    return {
        "pix_score": jnp.logaddexp(inlier_score, outlier_score),
        "inlier_score": inlier_score,
        "outlier_score": outlier_score,
    }


def threedp3_likelihood_per_pixel_old(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    filter_size,
):
    # filter_size = 3
    rendered_xyz_padded = jax.lax.pad(
        rendered_xyz,
        -100.0,
        (
            (
                filter_size,
                filter_size,
                0,
            ),
            (
                filter_size,
                filter_size,
                0,
            ),
            (
                0,
                0,
                0,
            ),
        ),
    )
    jj, ii = jnp.meshgrid(
        jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0])
    )
    indices = jnp.stack([ii, jj], axis=-1)
    log_probabilities = gausssian_mixture_vectorize_old(
        indices,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        filter_size,
    )
    return log_probabilities


def threedp3_gmm_likelihood(observed_rgbd, likelihood_args):
    variance = likelihood_args["variance"]
    outlier_prob = likelihood_args["outlier_prob"]
    outlier_volume = likelihood_args["outlier_volume"]
    filter_size = 3  # likelihood_args['filter_size']
    intrinsics = likelihood_args["intrinsics"]
    rendered_rgbd = likelihood_args["latent_rgbd"]

    observed_xyz = unproject_depth(observed_rgbd[..., 3], intrinsics)
    rendered_xyz = unproject_depth(rendered_rgbd[..., 3], intrinsics)

    log_probabilities_per_pixel = threedp3_likelihood_per_pixel_old(
        observed_xyz, rendered_xyz, variance, outlier_prob, outlier_volume, filter_size
    )

    return log_probabilities_per_pixel["pix_score"].sum(), log_probabilities_per_pixel


# threedp3_gmm_likelihood_vec = jax.jit(
#     jax.vmap(threedp3_gmm_likelihood, in_axes=(None, 0, None))
# )


def get_rgb_depth_inliers_from_observed_rendered_args(
    observed_rgb,
    rendered_rgb,
    observed_depth,
    rendered_depth,
    color_tolerance,
    depth_tolerance,
):
    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)
    error = jnp.linalg.norm(
        observed_lab[..., 1:3] - rendered_lab[..., 1:3], axis=-1
    ) + jnp.abs(observed_lab[..., 0] - rendered_lab[..., 0])

    valid_data_mask = rendered_rgb.sum(-1) != 0.0

    color_inliers = (error < color_tolerance) * valid_data_mask
    depth_inliers = (
        jnp.abs(observed_depth - rendered_depth) < depth_tolerance
    ) * valid_data_mask
    inliers = color_inliers * depth_inliers
    outliers = jnp.logical_not(inliers)
    teleport_outliers = (
        outliers * (observed_depth > rendered_depth) * (rendered_depth > 0.0)
    )
    nonteleport_outliers = outliers * jnp.logical_not(
        (observed_depth > rendered_depth) * (rendered_depth > 0.0)
    )
    return (
        inliers,
        color_inliers,
        depth_inliers,
        teleport_outliers,
        nonteleport_outliers,
        valid_data_mask,
    )


def kray_likelihood_intermediate(observed_rgbd, likelihood_args):
    rendered_rgbd = likelihood_args["latent_rgbd"]

    (
        inliers,
        color_inliers,
        depth_inliers,
        teleport_outliers,
        nonteleport_outliers,
        valid_data_mask,
    ) = get_rgb_depth_inliers_from_observed_rendered_args(
        observed_rgbd[..., :3],
        rendered_rgbd[..., :3],
        observed_rgbd[..., 3],
        rendered_rgbd[..., 3],
        likelihood_args["color_tolerance"],
        likelihood_args["depth_tolerance"],
    )

    outliers = teleport_outliers + nonteleport_outliers

    image_width, image_height, fx, fy, cx, cy, near, far = likelihood_args["intrinsics"]

    inlier_score = likelihood_args["inlier_score"]
    outlier_prob = likelihood_args["outlier_prob"]
    multiplier = likelihood_args["multiplier"]

    rendered_depth = rendered_rgbd[..., 3]
    observed_depth = observed_rgbd[..., 3]
    observed_depth_corrected = observed_depth + (observed_depth == 0.0) * far

    rendered_areas = (rendered_depth / fx) * (rendered_depth / fy)
    observed_areas = (observed_depth_corrected / fx) * (observed_depth_corrected / fy)

    A = 5
    inlier_contribution = (
        inlier_score * inliers * rendered_areas * (1 - outlier_prob) / A
    )

    teleport_factor = 0.00001
    V = 1 / 3 * jnp.power(far, 3) * image_width * image_height * 1 / (fx * fy)
    # V = 0.025
    outlier_contribution_teleporation = (
        teleport_outliers * observed_areas / V * teleport_factor * outlier_prob
    )
    outlier_contribution_not_teleportation = (
        nonteleport_outliers * observed_areas / V * outlier_prob
    )

    final_score_per_pix = (
        inlier_contribution
        + outlier_contribution_teleporation
        + outlier_contribution_not_teleportation
    )

    return jnp.log(jnp.sum(final_score_per_pix)) * multiplier, {
        "inliers": inliers,
        "color_inliers": color_inliers,
        "depth_inliers": depth_inliers,
        "outliers": outliers,
        # "undecided": undecided,
        "valid_data_mask": valid_data_mask,
        "inlier_contribution": inlier_contribution,
        "outlier_contribution": outlier_contribution_not_teleportation,
        "undecided_contribution": outlier_contribution_teleporation,
        "rendered_rgbd": rendered_rgbd,
        "pix_score": final_score_per_pix,
    }


# kray_likelihood_intermediate_vec = jax.jit(
#     jax.vmap(kray_likelihood_intermediate, in_axes=(None, 0, None))
# )
