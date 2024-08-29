import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree

import b3d


def raycast_to_image_nondeterministic(key, intrinsics, vertices_in_camera_frame, K):
    """
    Returns an array of shape (H, W, K) containing K point indices, or -1 to indicate no point was registered.
    """
    N_pts = vertices_in_camera_frame.shape[0]

    projected_pixel_coordinates = jnp.rint(
        b3d.xyz_to_pixel_coordinates(
            vertices_in_camera_frame,
            intrinsics["fx"],
            intrinsics["fy"],
            intrinsics["cx"],
            intrinsics["cy"],
        )
        - 0.5
    ).astype(jnp.int32)
    permutation = jax.random.permutation(key, N_pts)
    shuffled_pixel_coordinates = projected_pixel_coordinates[permutation]
    # shuffled_pixel_coordinates = projected_pixel_coordinates # = jax.random.permutation(key, projected_pixel_coordinates)

    random_indices = jax.random.randint(
        key, (N_pts,), 0, K
    )  # (N_pts,) array of random indices from 0 to K-1
    registered_pixel_indices = -jnp.ones(
        (intrinsics["height"], intrinsics["width"], K), dtype=int
    )
    registered_pixel_indices = registered_pixel_indices.at[
        shuffled_pixel_coordinates[:, 0],
        shuffled_pixel_coordinates[:, 1],
        random_indices,
    ].set(permutation)  # jnp.arange(N_pts))

    return registered_pixel_indices


@Pytree.dataclass
class PixelDistribution(genjax.ExactDensity):
    """
    registered_point_indices: (K,)
    all_rgbds: (N, 4)
    color_outlier_probs: (N,)
    depth_outlier_probs: (N,)
    color_scale: float
    depth_scale: float
    near: float
    far: float

    Where K is the max number of points registered at a pixel,
    N is the number of points in the scene.
    """

    def sample(
        key,
        registered_point_indices,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    ):
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        n_registered_points = jnp.sum(registered_point_indices != -1)
        idx = jax.random.randint(k1, (), 0, n_registered_points)
        depth_outlier_prob = jnp.where(
            n_registered_points == 0,
            1.0,
            depth_outlier_probs[registered_point_indices[idx]],
        )
        color_outlier_prob = jnp.where(
            n_registered_points == 0,
            1.0,
            color_outlier_probs[registered_point_indices[idx]],
        )
        uniform_depth_sample = jax.random.uniform(k2, (), minval=near, maxval=far)
        laplace_depth_sample = jax.scipy.stats.laplace.sample(
            k3, all_rgbds[registered_point_indices, 3], color_scale
        )
        uniform_rgb_sample = jax.random.uniform(k4, (3,), minval=0.0, maxval=1.0)
        laplace_rgb_sample = jax.scipy.stats.laplace.sample(
            k5, all_rgbds[registered_point_indices, :3], depth_scale
        )
        depth_is_outlier = jax.random.bernoulli(k6, depth_outlier_prob)
        color_is_outlier = jax.random.bernoulli(k7, color_outlier_prob)
        depth_sample = jnp.where(
            depth_is_outlier, uniform_depth_sample, laplace_depth_sample
        )
        rgb_sample = jnp.where(color_is_outlier, uniform_rgb_sample, laplace_rgb_sample)
        return jnp.concatenate([rgb_sample, depth_sample])

    def logpdf(
        obs,
        registered_point_indices,
        all_rgbds,
        color_outlier_probs,
        depth_outlier_probs,
        color_scale,
        depth_scale,
        near,
        far,
    ):
        uniform_depth_logpdf = -jnp.log(far - near)
        uniform_rgb_logpdf = -jnp.log(1.0**3)

        def get_logpdf_given_idx(idx):
            depth_outlier_prob = depth_outlier_probs[registered_point_indices[idx]]
            color_outlier_prob = color_outlier_probs[registered_point_indices[idx]]
            laplace_depth_logpdf = jax.scipy.stats.laplace.logpdf(
                obs[3], all_rgbds[registered_point_indices[idx], 3], color_scale
            )
            laplace_rgb_logpdf = jax.scipy.stats.laplace.logpdf(
                obs[:3], all_rgbds[registered_point_indices[idx], :3], depth_scale
            ).sum()
            log_p_depth = jnp.logaddexp(
                jnp.log(depth_outlier_prob) + uniform_depth_logpdf,
                jnp.log(1 - depth_outlier_prob) + laplace_depth_logpdf,
            )
            log_p_rgb = jnp.logaddexp(
                jnp.log(color_outlier_prob) + uniform_rgb_logpdf,
                jnp.log(1 - color_outlier_prob) + laplace_rgb_logpdf,
            )
            return log_p_depth + log_p_rgb

        n_registered_points = jnp.sum(registered_point_indices != -1)
        logpdfs_given_each_idx = jax.vmap(get_logpdf_given_idx)(
            n_registered_points.shape[0]
        )
        logpdf_of_choosing_each_idx = jnp.where(
            registered_point_indices == -1, -jnp.inf, -jnp.log(n_registered_points)
        )
        return jnp.where(
            n_registered_points > 0,
            jax.scipy.special.logsumexp(
                logpdfs_given_each_idx + logpdf_of_choosing_each_idx, axis=0
            ),
            uniform_depth_logpdf + uniform_rgb_logpdf,
        )


# @Pytree.dataclass
# class ImageDistribution(genjax.Distribution):
#     def random_weighted(key, args):
#         raycasted_image = raycast_to_image_nondeterministic(key, args)
#         value = mapped_pixel_distribution.sample(key, raycasted_image, args)
#         pdf_estimate = mapped_pixel_distribution.logpdf(value, raycasted_image, args)
#         return value, pdf_estimate

#     def estimate_logpdf(key, obs, args):
#         raycasted_image = raycast_to_image_nondeterministic(key, args)
#         pdf_estimate = mapped_pixel_distribution.logpdf(obs, raycasted_image, args)
#         return pdf_estimate
