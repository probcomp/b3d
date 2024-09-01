import importlib

import b3d
import b3d.chisight.dynamic_object_model.likelihoods.kfold_image_kernel as kik
import jax.numpy as jnp
from jax.random import PRNGKey

importlib.reload(kik)


def expected_logpdf_given_idx(args, value, point_idx):
    (
        _,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    ) = args
    i = point_idx

    color_pdf_should_be = jnp.logaddexp(
        jnp.log(1 - color_outlier_probs[i])
        + kik.truncated_color_laplace.logpdf(value[:3], all_rgbds[i][:3], color_scale),
        jnp.log(color_outlier_probs[i]) + jnp.log(1.0**3),
    )
    depth_pdf_should_be = jnp.logaddexp(
        jnp.log(1 - depth_outlier_probs[i])
        + kik.truncated_laplace.logpdf(
            value[3],
            all_rgbds[i][3],
            depth_scale,
            near,
            far,
            kik._FIXED_DEPTH_UNIFORM_WINDOW,
        ),
        jnp.log(depth_outlier_probs[i]) + jnp.log(1 / (far - near)),
    )
    return color_pdf_should_be + depth_pdf_should_be


def test_logpdfs_in_image_with_one_point_per_pixel():
    intrinsics = (
        3,
        3,
        200.0,
        200.0,
        50.0,
        50.0,
        0.01,
        10.0,
    )
    image_width, image_height, fx, fy, cx, cy, _, _ = intrinsics

    intrinsics_dict = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": image_width,
        "height": image_height,
        "near": 0.01,
        "far": 100.0,
    }
    depth_image = jnp.ones((image_height, image_width), dtype=jnp.float32)
    vertices = b3d.camera.camera_from_depth(depth_image, intrinsics).reshape(-1, 3)

    # result = kik.raycast_to_image_nondeterministic(
    #     PRNGKey(0),
    #     {
    #         "height": image_height,
    #         "width": image_width,
    #         "fx": fx,
    #         "fy": fy,
    #         "cx": cx,
    #         "cy": cy,
    #     },
    #     vertices,
    #     2,
    # )
    # result

    rgbds = jnp.tile(jnp.array([1.0, 0.0, 0.0, 5.0]), (9, 1))
    color_outlier_probs = 0.003 * jnp.arange(9)
    depth_outlier_probs = (10 - jnp.arange(9)) / 100
    color_scale = 0.01
    depth_scale = 0.04

    image_kernel = kik.KfoldMixturePointsToImageKernel(1)
    sample, lp1 = image_kernel.random_weighted(
        PRNGKey(0),
        intrinsics_dict,
        vertices,
        rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
    )
    lp2 = image_kernel.estimate_logpdf(
        PRNGKey(10),
        sample,
        intrinsics_dict,
        vertices,
        rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
    )
    assert lp1 == lp2
    expected_lp = sum(
        [
            expected_logpdf_given_idx(
                (
                    None,
                    rgbds,
                    color_outlier_probs,
                    depth_outlier_probs,
                    color_scale,
                    depth_scale,
                    intrinsics_dict["near"],
                    intrinsics_dict["far"],
                ),
                sample[*jnp.unravel_index(i, (3, 3))],
                i,
            )
            for i in range(9)
        ]
    )
    assert lp1 == expected_lp

    test_image = jnp.tile(jnp.array([1.0, 0.0, 0.0, 5.0]), (3, 3, 1))
    test_image_2 = jnp.tile(jnp.array([1.0, 0.0, 1.0, 5.0]), (3, 3, 1))
    lp3 = image_kernel.estimate_logpdf(
        PRNGKey(10),
        test_image,
        intrinsics_dict,
        vertices,
        rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
    )
    lp4 = image_kernel.estimate_logpdf(
        PRNGKey(10),
        test_image_2,
        intrinsics_dict,
        vertices,
        rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
    )
    assert lp3 > lp4


# TODO: we could do more thorough testing.
