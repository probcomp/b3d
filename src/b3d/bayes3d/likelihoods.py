import jax
import jax.numpy as jnp

import b3d


class PixelOutlier:
    @staticmethod
    def get_rgb_depth_inliers_from_observed_rendered_args(
        observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args
    ):
        observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
        rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)
        error = jnp.linalg.norm(
            observed_lab[..., 1:3] - rendered_lab[..., 1:3], axis=-1
        ) + jnp.abs(observed_lab[..., 0] - rendered_lab[..., 0])

        valid_data_mask = rendered_rgb.sum(-1) != 0.0

        color_inliers = (error < model_args.color_tolerance) * valid_data_mask
        depth_inliers = (
            jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance
        ) * valid_data_mask
        inliers = color_inliers * depth_inliers
        outliers = jnp.logical_not(inliers) * valid_data_mask
        undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)
        return (
            inliers,
            color_inliers,
            depth_inliers,
            outliers,
            undecided,
            valid_data_mask,
        )

    @staticmethod
    def logpdf(
        observed_rgb,
        observed_depth,
        rendered_rgb,
        rendered_depth,
        fx,
        fy,
        height,
        width,
        near,
        far,
        model_args,
    ):
        (
            inliers,
            _color_inliers,
            _depth_inliers,
            outliers,
            undecided,
            _valid_data_mask,
        ) = PixelOutlier.get_rgb_depth_inliers_from_observed_rendered_args(
            observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args
        )

        inlier_weight = model_args.inlier_score
        outlier_prob = model_args.outlier_prob
        multiplier = model_args.color_multiplier

        corrected_depth = rendered_depth + (rendered_depth == 0.0) * far
        areas = (corrected_depth / fx) * (corrected_depth / fy)

        inlier_score = inlier_weight * jnp.sum(inliers * areas)
        undecided_score = 1.0 * jnp.sum(undecided * areas)
        outlier_score = outlier_prob * jnp.sum(outliers * areas)

        final_log_score = (
            jnp.log(
                # This is leaving out a 1/A (which does depend upon the scene)
                inlier_score + undecided_score + outlier_score
            )
            * multiplier
        )

        return {
            "log_score": final_log_score,
            "inlier_score": inlier_score,
            "undecided_score": undecided_score,
            "outlier_score": outlier_score,
        }


pixel_outlier_logpdfs = jax.vmap(
    PixelOutlier.logpdf, (None, None, 0, 0, None, None, None, None, None, None, None)
)
