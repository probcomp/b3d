import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from tensorflow_probability.substrates import jax as tfp

import b3d
from b3d.chisight.dense.likelihoods.other_likelihoods import (
    ImageDistFromPixelDist,
)


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
class TruncatedLaplace(genjax.ExactDensity):
    """
    This is a distribution on the interval (low, high).
    The generative process is:
    1. Sample x ~ laplace(loc, scale).
    2. If x < low, sample y ~ uniform(low, low + uniform_window_size) and return y.
    3. If x > high, sample y ~ uniform(high - uniform_window_size, high) and return y.
    4. Otherwise, return x.

    Args:
    - loc: float
    - scale: float
    - low: float
    - high: float
    - uniform_window_size: float

    Support:
    - x in (low, high) [a float]
    """

    def sample(self, key, loc, scale, low, high, uniform_window_size):
        # assert low < high
        # assert low + uniform_window_size < high - uniform_window_size
        k1, k2 = jax.random.split(key, 2)
        x = tfp.distributions.Laplace(loc, scale).sample(seed=k1)
        u = jax.random.uniform(k2, ()) * uniform_window_size
        return jnp.where(
            x > high, high - uniform_window_size + u, jnp.where(x < low, low + u, x)
        )

    def logpdf(self, obs, loc, scale, low, high, uniform_window_size):
        assert low < high
        assert low + uniform_window_size < high - uniform_window_size
        laplace_logpdf = tfp.distributions.Laplace(loc, scale).log_prob(obs)
        laplace_p_below_low = tfp.distributions.Laplace(loc, scale).cdf(low)
        laplace_p_above_high = 1 - tfp.distributions.Laplace(loc, scale).cdf(high)

        return jnp.where(
            jnp.logical_and(
                low + uniform_window_size < obs, obs < high - uniform_window_size
            ),
            laplace_logpdf,
            jnp.where(
                obs < low + uniform_window_size,
                jnp.logaddexp(
                    jnp.log(laplace_p_below_low / uniform_window_size), laplace_logpdf
                ),
                jnp.logaddexp(
                    jnp.log(laplace_p_above_high / uniform_window_size), laplace_logpdf
                ),
            ),
        )


truncated_laplace = TruncatedLaplace()


_FIXED_COLOR_UNIFORM_WINDOW = 1 / 255
_FIXED_DEPTH_UNIFORM_WINDOW = 0.01


@Pytree.dataclass
class TruncatedColorLaplace(genjax.ExactDensity):
    """
    Args:
    - loc: (3,) array (loc for R, G, B channels)
    - shared_scale: float (scale, shared across R, G, B channels)
    - uniform_window_size: float [optional; defaults to 1/255]

    Support:
    - rgb in [0, 1]^3 [a 3D array]
    """

    def sample(
        self, key, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda k, lc: truncated_laplace.sample(
                k, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(jax.random.split(key, loc.shape[0]), loc)

    def logpdf(
        self, obs, loc, shared_scale, uniform_window_size=_FIXED_COLOR_UNIFORM_WINDOW
    ):
        return jax.vmap(
            lambda o, lc: truncated_laplace.logpdf(
                o, lc, shared_scale, 0.0, 1.0, uniform_window_size
            ),
            in_axes=(0, 0),
        )(obs, loc).sum()


truncated_color_laplace = TruncatedColorLaplace()


def _access(arr, idx):
    return jax.lax.dynamic_index_in_dim(arr, idx, axis=0, keepdims=False)


@Pytree.dataclass
class PixelDistribution(genjax.ExactDensity):
    """
    Distribution over the color observed at a pixel of an RGBD image,
    given a set of points that may be registered at the pixel.
    Each of the N points has an associated color_outlier_prob,
    depth_outlier_prob, and RGBD value.
    There is a global color_scale and depth_scale.
    An array `registered_point_indices` of shape (K,)
    is provided giving the indices of all the points registered at this pixel;
    the value -1 indicates that no point is registered at this slot of the
    `registered_point_indices` array.

    Args:
        registered_point_indices: (K,)
        all_rgbds: (N, 4)
        color_outlier_probs: (N,)
        depth_outlier_probs: (N,)
        color_scale: float
        depth_scale: float
        near: float
        far: float

    Support:
    - `rgbd` in [0, 1]^3 x [near, far] [a 4D array]

    K is the max number of points registered at a pixel, and
    N is the number of points in the scene.
    Indices in registered_point_indices are in the range [-1, N-1];
    -1 indicates that no point is registered in this slot.

    The generative process is:
    1. If there are no registered points, sample uniformly from the color and depth ranges.
    2. Otherwise, sample an index idx from registered_point_indices s.t. idx > -1.
    3. Get the rgbd, color_outlier_prob, and depth_outlier_prob for this index.
    4. Sample is_depth_outlier ~ Bernoulli(depth_outlier_prob).
    5. Sample is_color_outlier ~ Bernoulli(color_outlier_prob).
    6. If is_depth_outlier, sample depth uniformly from [near, far].
    7. Otherwise, sample depth from a truncated Laplace distribution centered at all_rgbds[idx, 3].
    8. If is_color_outlier, sample color uniformly from [0, 1]^3.
    9. Otherwise, sample color from a truncated Laplace distribution centered at all_rgbds[idx, :3].

    The generative process for the truncated Laplace distribuitons
    sample from a Laplace, and if the sample is outside the range [low, high]
    (which is either [near, far] for depth or [0, 1] for color), the sample is replaced with a uniform sample
    from the range [low, low + uniform_window_size] or [high - uniform_window_size, high].
    The value `uniform_window_size` is currently fixed at 0.01 for depth and 1/255 for color.
    """

    def sample(
        self,
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
        idxprobs = jnp.where(
            registered_point_indices >= 0, 1.0 / n_registered_points, 0.0
        )
        idx = genjax.categorical.sample(k1, jnp.log(idxprobs))
        depth_outlier_prob = jnp.where(
            n_registered_points == 0,
            1.0,
            _access(depth_outlier_probs, _access(registered_point_indices, idx)),
        )
        color_outlier_prob = jnp.where(
            n_registered_points == 0,
            1.0,
            _access(color_outlier_probs, _access(registered_point_indices, idx)),
        )
        uniform_depth_sample = jax.random.uniform(k2, (), minval=near, maxval=far)
        laplace_depth_sample = truncated_laplace.sample(
            k3,
            _access(all_rgbds, _access(registered_point_indices, idx))[3],
            depth_scale,
            near,
            far,
            _FIXED_DEPTH_UNIFORM_WINDOW,
        )
        uniform_rgb_sample = jax.random.uniform(k4, (3,), minval=0.0, maxval=1.0)
        laplace_rgb_sample = truncated_color_laplace.sample(
            k5,
            _access(all_rgbds, _access(registered_point_indices, idx))[:3],
            color_scale,
        )
        depth_is_outlier = jax.random.bernoulli(k6, depth_outlier_prob)
        color_is_outlier = jax.random.bernoulli(k7, color_outlier_prob)
        depth_sample = jnp.where(
            depth_is_outlier, uniform_depth_sample, laplace_depth_sample
        )
        rgb_sample = jnp.where(color_is_outlier, uniform_rgb_sample, laplace_rgb_sample)
        return jnp.concatenate([rgb_sample, jnp.array([depth_sample])])

    def logpdf(
        self,
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
            depth_outlier_prob = _access(
                depth_outlier_probs, _access(registered_point_indices, idx)
            )
            color_outlier_prob = _access(
                color_outlier_probs, _access(registered_point_indices, idx)
            )
            laplace_depth_logpdf = truncated_laplace.logpdf(
                obs[3],
                _access(all_rgbds, _access(registered_point_indices, idx))[3],
                depth_scale,
                near,
                far,
                _FIXED_DEPTH_UNIFORM_WINDOW,
            )
            laplace_rgb_logpdf = truncated_color_laplace.logpdf(
                obs[:3],
                _access(all_rgbds, _access(registered_point_indices, idx))[:3],
                color_scale,
            )
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
            jnp.arange(registered_point_indices.shape[0])
        )
        logpdf_of_choosing_each_idx = jnp.where(
            registered_point_indices < 0, -jnp.inf, -jnp.log(n_registered_points)
        )
        assert len((logpdf_of_choosing_each_idx + logpdfs_given_each_idx).shape) == 1
        return jnp.where(
            n_registered_points > 0,
            jax.scipy.special.logsumexp(
                logpdf_of_choosing_each_idx + logpdfs_given_each_idx
            ),
            uniform_depth_logpdf + uniform_rgb_logpdf,
        )


pixel_distribution = PixelDistribution()

mapped_pixel_distribution = ImageDistFromPixelDist(
    # The only mapped arg is the first one; the others are shared across pixels.
    pixel_distribution,
    (True, False, False, False, False, False, False, False),
)
# ^ This distribution accepts (height, width, registered_pixel_indices, *args),
# where `registered_pixel_indices` has shape (height, width, K),
# and `args` is the list of all args to PixelDistribution after `registered_pixel_indices`.


@Pytree.dataclass
class KfoldMixturePointsToImageKernel(genjax.Distribution):
    """
    KfoldMixturePointsToImageKernel(K) is a kernel from a set of points to an image.
    Each pixel in the image will register a random subset of up to K points from the
    subset of the provided points which project directly to that pixel.

    Given those up to K registered points per pixel, each pixel is independently
    sampled from `PixelDistribution` (see docstring for `PixelDistribution` for details).

    Constructor args:
    - K: int

    Distribution args:
    - intrinsics: dict with keys "height", "width", "fx", "fy", "cx", "cy", "near", "far"
    - vertices_in_camera_frame: (N, 3) array of points in camera frame
    - point_rgbds: (N, 4) array of RGBD values for each point
    - point_color_outlier_probs: (N,) array of color outlier probabilities for each point
    - point_depth_outlier_probs: (N,) array of depth outlier probabilities for each point
    - color_scale: float (shared)
    - depth_scale: float (shared)

    Distribution support:
    - image: (height, width, 4) array of RGBD values (RGB values in [0, 1]^3; D values in [near, far])
    """

    K: int

    def __init__(self, K):
        self.K = K

    def random_weighted(
        self,
        key,
        intrinsics,
        vertices_in_camera_frame,
        point_rgbds,
        point_color_outlier_probs,
        point_depth_outlier_probs,
        color_scale,
        depth_scale,
    ):
        h, w = intrinsics["height"], intrinsics["width"]
        raycasted_image = raycast_to_image_nondeterministic(
            key, intrinsics, vertices_in_camera_frame, self.K
        )
        value = mapped_pixel_distribution.sample(
            key,
            h,
            w,
            raycasted_image,
            point_rgbds,
            point_color_outlier_probs,
            point_depth_outlier_probs,
            color_scale,
            depth_scale,
            intrinsics["near"],
            intrinsics["far"],
        )
        logpdf_estimate = mapped_pixel_distribution.logpdf(
            value,
            h,
            w,
            raycasted_image,
            point_rgbds,
            point_color_outlier_probs,
            point_depth_outlier_probs,
            color_scale,
            depth_scale,
            intrinsics["near"],
            intrinsics["far"],
        )
        return value, logpdf_estimate

    def estimate_logpdf(
        self,
        key,
        obs,
        intrinsics,
        vertices_in_camera_frame,
        point_rgbds,
        point_color_outlier_probs,
        point_depth_outlier_probs,
        color_scale,
        depth_scale,
    ):
        h, w = intrinsics["height"], intrinsics["width"]
        raycasted_image = raycast_to_image_nondeterministic(
            key, intrinsics, vertices_in_camera_frame, self.K
        )
        logpdf_estimate = mapped_pixel_distribution.logpdf(
            obs,
            h,
            w,
            raycasted_image,
            point_rgbds,
            point_color_outlier_probs,
            point_depth_outlier_probs,
            color_scale,
            depth_scale,
            intrinsics["near"],
            intrinsics["far"],
        )
        return logpdf_estimate
