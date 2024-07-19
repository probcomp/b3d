
import genjax
from genjax.generative_functions.distributions import ExactDensity
import jax.numpy as jnp
import b3d
from b3d import Mesh, Pose
from collections import namedtuple
from b3d.modeling_utils import uniform_discrete, uniform_pose
import jax
import os

from genjax import Pytree


def get_rgb_depth_inliers_from_observed_rendered_args(
        observed_rgb,
        rendered_rgb,
        observed_depth,
        rendered_depth,
        color_tolerance,
        depth_tolerance

    ):
    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)
    error = (
        jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1)
        # + jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
    )

    valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

    color_inliers = (error < color_tolerance) * valid_data_mask
    depth_inliers = (jnp.abs(observed_depth - rendered_depth) < depth_tolerance) * valid_data_mask
    inliers = color_inliers * depth_inliers
    outliers = jnp.logical_not(inliers) * valid_data_mask
    undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)
    return (inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask)


def kray_likelihood_intermediate(observed_rgbd, scene_mesh, renderer, likelihood_args):
    rendered_rgbd = renderer.render_rgbd(
        scene_mesh.vertices,
        scene_mesh.faces,
        scene_mesh.vertex_attributes,
    )
    (inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask) = get_rgb_depth_inliers_from_observed_rendered_args(
        observed_rgbd[...,:3],
        rendered_rgbd[...,:3],
        observed_rgbd[...,3],
        rendered_rgbd[...,3],
        likelihood_args["color_tolerance"],
        likelihood_args["depth_tolerance"]
    )

    fx = renderer.fx
    fy = renderer.fy
    inlier_score = likelihood_args["inlier_score"]
    outlier_prob = likelihood_args["outlier_prob"]
    multiplier = likelihood_args["multiplier"]

    rendered_depth = rendered_rgbd[...,3]
    observed_depth = observed_rgbd[...,3]
    observed_depth_corrected = observed_depth + (observed_depth == 0.0) * renderer.far

    rendered_areas = (rendered_depth / fx) * (rendered_depth / fy)
    observed_areas = (observed_depth_corrected / fx) * (observed_depth_corrected / fy)

    inlier_contribution = jnp.sum(inlier_score *  inliers * rendered_areas)
    V = 1/3 * jnp.power(renderer.far, 3) * renderer.width * renderer.height * 1/(fx * fy)
    outlier_contribution_teleporation = jnp.sum(outliers * observed_areas / V * 0.001 * outlier_prob * (observed_depth > rendered_depth))
    outlier_contribution_not_teleportation = jnp.sum(outliers * observed_areas / V * outlier_prob * (observed_depth <= rendered_depth))

    final_log_score =  jnp.log(inlier_contribution + outlier_contribution_teleporation + outlier_contribution_not_teleportation) * multiplier

    return {
        "inliers": inliers,
        "color_inliers": color_inliers,
        "depth_inliers": depth_inliers,
        "outliers": outliers,
        "undecided": undecided,
        "valid_data_mask": valid_data_mask,
        "inlier_contribution": inlier_contribution,
        "outlier_contribution": outlier_contribution_not_teleportation,
        "undecided_contribution": outlier_contribution_teleporation,
        "rendered_rgbd": rendered_rgbd,
        "score": final_log_score,
    }
