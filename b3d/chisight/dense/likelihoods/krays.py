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


KRaysImageLikelihoodArgs = namedtuple('KRaysImageLikelihoodArgs', [
    'color_tolerance',
    'depth_tolerance',
    'inlier_score',
    'outlier_prob',
    'multiplier',
])


def get_rgb_depth_inliers_from_observed_rendered_args(observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args):
    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)
    error = (
        jnp.linalg.norm(observed_lab[...,1:3] - rendered_lab[...,1:3], axis=-1) +
        jnp.abs(observed_lab[...,0] - rendered_lab[...,0])
    )

    valid_data_mask = (rendered_rgb.sum(-1) != 0.0)

    color_inliers = (error < model_args.color_tolerance) * valid_data_mask
    depth_inliers = (jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance) * valid_data_mask
    inliers = color_inliers * depth_inliers
    outliers = jnp.logical_not(inliers) * valid_data_mask
    undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)
    return (inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask)



def make_krays_image_observation_model(renderer):
    @Pytree.dataclass
    class KRaysImageLikelihood(genjax.ExactDensity):
        def sample(self, key, rendered_rgbd, likelihood_args):
            return rendered_rgbd

        def logpdf(self, observed_rgbd, rendered_rgbd, likelihood_args):
            inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask = get_rgb_depth_inliers_from_observed_rendered_args(
                observed_rgbd[...,:3],
                rendered_rgbd[...,:3],
                observed_rgbd[...,3],
                rendered_rgbd[...,3],
                likelihood_args
            )

            inlier_score = likelihood_args.inlier_score
            outlier_prob = likelihood_args.outlier_prob
            multiplier = likelihood_args.multiplier

            corrected_depth = rendered_rgbd[...,3] + (rendered_rgbd[...,3] == 0.0) * renderer.far
            areas = (corrected_depth / renderer.fx) * (corrected_depth / renderer.fy)

            return jnp.log(
                # This is leaving out a 1/A (which does depend upon the scene)
                inlier_score * jnp.sum(inliers * areas) +
                1.0 * jnp.sum(undecided * areas)  +
                outlier_prob * jnp.sum(outliers * areas)
            ) * multiplier

    krays_image_likelihood = KRaysImageLikelihood()

    @genjax.gen
    def krays_image_observation_model(mesh, likelihood_args):
        noiseless_rendered_rgbd = renderer.render_rgbd(mesh.vertices, mesh.faces, mesh.vertex_attributes)
        image = krays_image_likelihood(noiseless_rendered_rgbd, likelihood_args) @ "image"
        return (image, noiseless_rendered_rgbd)

    return krays_image_observation_model
