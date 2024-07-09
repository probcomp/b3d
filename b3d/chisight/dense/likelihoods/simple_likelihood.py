import jax.numpy as jnp
import jax
import b3d

def simple_likelihood(observed_rgbd, scene_mesh, renderer, likelihood_args):
    rendered_rgbd = renderer.render_rgbd(
        scene_mesh.vertices,
        scene_mesh.faces,
        scene_mesh.vertex_attributes,
    )

    fx = renderer.fx
    fy = renderer.fy
    far = renderer.far

    rendered_rgb = rendered_rgbd[..., :3]
    observed_rgb = observed_rgbd[..., :3]

    rendered_depth = rendered_rgbd[...,3]
    observed_depth = observed_rgbd[...,3]

    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)

    rendered_areas = (rendered_depth / fx) * (rendered_depth / fy)

    is_hypothesized = (rendered_depth > 0.0)
    
    is_observed_data_rgb = (observed_rgb.min(-1) < 0.99) * (observed_rgb.max(-1) > 0.01)
    is_observed_data_depth = (observed_depth > 0.0)

    bounds = likelihood_args["bounds"]

    color_match = (jnp.abs(observed_lab - rendered_lab) < bounds[:3]).all(-1) * is_hypothesized * is_observed_data_rgb
    depth_match = (jnp.abs(observed_depth - rendered_depth) < bounds[3]) * is_hypothesized * is_observed_data_depth

    is_match = color_match * depth_match
    is_color_matched_but_no_depth_data = color_match * ~is_observed_data_depth
    is_mismatched = (
        is_hypothesized * ~is_match * ~is_color_matched_but_no_depth_data * is_observed_data_rgb * is_observed_data_depth
    )


    score = jnp.sum(
        is_match * rendered_areas * 2.0 + is_mismatched * rendered_areas * -1.0
    ) * likelihood_args["multiplier"]

    return {
        "score": score,
        "is_match": is_match,
        "is_mismatched": is_mismatched,
        "color_match": color_match,
        "depth_match": depth_match,
        "is_hypothesized": is_hypothesized, 
        "rendered_rgbd": rendered_rgbd,     
    }
