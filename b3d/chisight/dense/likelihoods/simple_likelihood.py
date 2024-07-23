import jax.numpy as jnp
import jax
import b3d


@jax.jit
def simple_likelihood(observed_rgbd, rendered_rgbd, likelihood_args):
    fx = likelihood_args["fx"]
    fy = likelihood_args["fy"]

    rendered_rgb = rendered_rgbd[..., :3]
    observed_rgb = observed_rgbd[..., :3]

    rendered_depth = rendered_rgbd[..., 3]
    observed_depth = observed_rgbd[..., 3]

    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)

    rendered_areas = (rendered_depth / fx) * (rendered_depth / fy)

    is_hypothesized = rendered_depth > 0.0

    is_observed_data_rgb = (observed_rgb.min(-1) < 0.99) * (observed_rgb.max(-1) > 0.01)
    is_observed_data_depth = observed_depth > 0.0

    bounds = likelihood_args["bounds"]

    color_match = (
        (jnp.abs(observed_lab - rendered_lab) < bounds[:3]).all(-1)
        * is_hypothesized
        * is_observed_data_rgb
    )
    depth_match = (
        (jnp.abs(observed_depth - rendered_depth) < bounds[3])
        * is_hypothesized
        * is_observed_data_depth
    )
    is_match = color_match * depth_match

    is_mismatched = (
        is_hypothesized * ~is_match * is_observed_data_depth * is_observed_data_rgb
    )

    is_mismatched_teleportation = is_mismatched * (rendered_depth < observed_depth)
    is_mismatched_non_teleportation = is_mismatched * ~is_mismatched_teleportation

    score = (
        jnp.sum(
            is_match * rendered_areas * 6.0
            + is_mismatched_non_teleportation * rendered_areas * -1.0
            + is_mismatched_teleportation * rendered_areas * -2.0
        )
        * likelihood_args["multiplier"]
    )

    return {
        "score": score,
        "is_match": is_match,
        "is_mismatched": is_mismatched_non_teleportation,
        "is_mismatched_teleportation": is_mismatched_teleportation,
        "color_match": color_match,
        "depth_match": depth_match,
        "is_hypothesized": is_hypothesized,
        "rendered_rgbd": rendered_rgbd,
        "alternate_color_space": observed_lab,
        "alternate_color_spcae_rendered": rendered_lab,
    }
