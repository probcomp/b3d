import jax
import jax.numpy as jnp
import genjax
import genjax.typing
from genjax import ExactDensity
from b3d.utils import keysplit
from b3d.camera import (
    camera_from_screen_and_depth,
    screen_from_camera,
)
from b3d.types import Array, Matrix, Float
from jax.scipy.spatial.transform import Rotation as Rot
from .pose_utils import (
    uniform_samples_from_disc,
)
from .dynamic_gps import DynamicGPS
from typing import TypeAlias


inv = jnp.linalg.inv
logsumexp = jax.scipy.special.logsumexp
logaddexp = jnp.logaddexp
CovarianceMatrix: TypeAlias = Matrix
PrecisionMatrix: TypeAlias = Matrix
RayOrigin: TypeAlias = Array
RayDirection: TypeAlias = Array


def ellipsoid_embedding(cov: CovarianceMatrix) -> Matrix:
    """Returns A with cov = A@A.T"""
    sigma, U = jnp.linalg.eigh(cov)
    D = jnp.diag(jnp.sqrt(sigma))
    return U @ D @ U.T


def dq_from_cov(cov: CovarianceMatrix):
    sigma, U = jnp.linalg.eigh(cov)
    q = Rot.from_matrix(U).as_quat()
    return sigma, q


def bilinear(x: Array, y: Array, A: Matrix) -> Float:
    return x.T @ A @ y


def gaussian_restriction_to_ray(
    mu_tilde, prec: PrecisionMatrix, o: RayOrigin, v: RayDirection
):
    """
    Restricts a normalized Gaussian to a ray and returns
    the mean `mu` and standard deviation `sig`, such that,
    parameterizing they ray by $r(t) = o +t*v$, we have
    $$
        N( r(t) | \tilde\mu, cov) = w * N(t | \mu, \sigma)
    $$
    where
    $$
        w = N( r(\mu) | \tilde\mu, cov) /  N(\mu, \mu, \sigma).
    $$
    Note that $\mu$ is the maximum of both the nominator and denominator.
    Also note that the first equation implies that the integral of
    the Gaussian along the ray is given by $w$.
    """
    mu1D = bilinear(mu_tilde - o, v, prec) / bilinear(v, v, prec)
    sig1D = 1 / jnp.sqrt(bilinear(v, v, prec))
    return mu1D, sig1D


def cov_from_dq_composition(diag, quat):
    """
    Covariance matrix from particle representation `(diag, quat)`,
    where `diag` is an array of eigenvalues and `quat` is a quaternion
    representing the matrix of eigenvectors.
    """
    U = Rot.from_quat(quat).as_matrix()
    C = U @ jnp.diag(diag) @ U.T
    return C


# TODO: Test this code
# TODO: Add constraint for the point light to fall within image bounds
@genjax.Pytree.dataclass
class ProjectiveGaussian(ExactDensity):
    def sample(self, key, mu, cov, cam, intr):
        """
        Samples a 2d pointlight on the sensor canvas from a 3d Gaussian distribution.
        """
        x = jax.random.multivariate_normal(key, mu, cov)
        uv = screen_from_camera(cam.inv().apply(x), intr)
        return uv

    def logpdf(self, uv, mu, cov, cam, intr):
        """
        Evaluates the log probability of a 2d pointlight
        under a 3d Gaussian distribution.
        """
        prec = inv(cov)
        o = cam.pos
        x = cam.apply(camera_from_screen_and_depth(uv, jnp.array(1.0), intr))
        v = x - o

        # The mode of the restriction of the Gaussian to the ray
        # from unprojecting the 2d point light at uv.
        t, sig = gaussian_restriction_to_ray(mu, prec, o, v)

        # The likelihood of the sensor measurement is given by the
        # integral of the Gaussian along the ray; see the
        # docstring of `gaussian_restriction_to_ray`.
        logp = (
            sig
            * jnp.sqrt(2 * jnp.pi)
            * jax.scipy.stats.multivariate_normal.logpdf(o + t * v, mu, cov)
        )

        return logp


projective_gaussian = ProjectiveGaussian()


# TODO: Test this code
@genjax.Pytree.dataclass
class ProjectiveGaussianMixture(ExactDensity):
    def sample(self, key, log_weights, mus, covs, cam, intr):
        _, keys = keysplit(key, 1, 2)
        i = jax.random.categorical(keys[0], log_weights)
        jbinder = (  # noqa: E731
            lambda j: lambda key, mus, covs, cam, intr: projective_gaussian.sample(
                key, mus[j], covs[j], cam, intr
            )
        )
        branches = [jbinder(j) for j in jnp.arange(log_weights.shape[0])]
        x = jax.lax.switch(i, branches, keys[1], mus, covs, cam, intr)
        return x

    def logpdf(self, x, log_weights, mus, covs, cam, intr):
        logps = jax.vmap(projective_gaussian.logpdf, (None, 0, 0, None, None))(
            x, mus, covs, cam, intr
        )
        normalized_log_weights = log_weights - logsumexp(log_weights)
        logp = logsumexp(logps + normalized_log_weights)
        return logp


projective_gaussian_mixture = ProjectiveGaussianMixture()


# TODO: Test this code
@genjax.Pytree.dataclass
class HomogeneousMixture(ExactDensity):
    dist: genjax.typing.Any

    def sample(self, key, log_weights, comp_args):
        """
        Args:
            `key`: PRNGKey
            `log_weights`: Log weights of the components
            `comp_args`: Arguments for the components
                (each row corresponds to a component)
        """
        _, keys = keysplit(key, 1, 2)
        i = jax.random.categorical(keys[0], log_weights)
        jbinder = lambda j: lambda: self.dist.sample(keys[1], *comp_args[i])  # noqa: E731
        branches = [jbinder(j) for j in jnp.arange(log_weights.shape[0])]
        x = jax.lax.switch(i, branches)
        return x

    def logpdf(self, x, log_weights, comp_args):
        """
        Args:
            `x`: Sample from mixture
            `log_weights`: Log weights of the components
            `comp_args`: Arguments for the components
                (each row corresponds to a component)
        """
        logps = jax.vmap(lambda args: self.dist.logpdf(x, *args))(comp_args)
        normalized_log_weights = log_weights - logsumexp(log_weights)
        logp = jnp.logsumexp(logps + normalized_log_weights)
        return logp


# TODO: Test this code
@genjax.Pytree.dataclass
class TwoComponentMixture(ExactDensity):
    p0: genjax.typing.Any
    p1: genjax.typing.Any

    def sample(self, key, log_weights, comp_args):
        """
        Args:
            `key`: PRNGKey
            `log_weights`: Log weights of the components
            `comp_args`: Tuple of arguments for each components
        """
        _, keys = keysplit(key, 1, 2)
        i = jax.random.categorical(keys[0], log_weights)
        x = jax.lax.switch(
            i,
            [
                lambda: self.p0.sample(keys[1], *comp_args[0]),
                lambda: self.p1.sample(keys[1], *comp_args[1]),
            ],
        )
        return x

    def logpdf(self, x, log_weights, comp_args):
        """
        Args:
            `x`: Sample from mixture
            `log_weights`: Log weights of the components
            `comp_args`: Tuple of arguments for each components
        """
        logp = jnp.logaddexp(
            log_weights[0] + self.p0.logpdf(x, *comp_args[0]),
            log_weights[1] + self.p1.logpdf(x, *comp_args[1]),
        )
        return logp


@genjax.Pytree.dataclass
class IndexDist(ExactDensity):
    """
    Distribution over arrival indices conditioned on being an outlier or not.
    We mimic a masking combinator here, we don't want to score the arrival index
    in the case of being an outlier.
    """

    def sample(self, key, is_outlier, logprobs):
        i = jax.random.categorical(key, logprobs)
        return jnp.where(is_outlier, -1, i)

    def logpdf(self, i, is_outlier, logprobs):
        return jnp.where(is_outlier, 0.0, logprobs[i])


index_dist = IndexDist()

@genjax.Pytree.dataclass
class MixtureHack(ExactDensity):
    def sample(self, key, is_outlier, i, mus, covs, cam, intr):
        _, keys = keysplit(key, 1, 2)
        outlier = jax.random.uniform(
            keys[0],
            minval=jnp.zeros(2),
            maxval=jnp.array([intr.width, intr.height], dtype=jnp.float32),
            shape=(2,),
        )
        inlier = projective_gaussian.sample(keys[1], mus[i], covs[i], cam, intr)
        return jnp.where(is_outlier, outlier, inlier)

    def logpdf(self, uv, is_outlier, i, mus, covs, cam, intr):
        outlier_logp = -jnp.log(intr.width * intr.height)
        inlier_logp = projective_gaussian.logpdf(uv, mus[i], covs[i], cam, intr)
        in_bounds = (uv[0] <= intr.width) * (uv[1] <= intr.height)

        return jnp.where(
            in_bounds, jnp.where(is_outlier, outlier_logp, inlier_logp), -jnp.inf
        )


mixture_hack = MixtureHack()


# TODO: Test this code
@genjax.Pytree.dataclass
class MixtureStepHack(ExactDensity):
    def sample(self, key, is_outlier, i, mus, covs, cam, intr):
        _, keys = keysplit(key, 1, 2)
        # Sample from the outlier distribution
        outlier = jax.random.uniform(
            keys[0],
            minval=jnp.zeros(2),
            maxval=jnp.array([intr.width, intr.height], dtype=jnp.float32),
            shape=(2,),
        )

        # Sample from the inlier distribution
        A = ellipsoid_embedding(covs[i])
        x = uniform_samples_from_disc(keys[1], 1, d=3)[0]
        inlier = screen_from_camera(cam.inv().apply(A @ x + mus[i]), intr)
        return jnp.where(is_outlier, outlier, inlier)

    def logpdf(self, uv, is_outlier, i, mus, covs, cam, intr):
        # The Gaussian parameters
        cov = covs[i]
        prec = inv(cov)
        mu = mus[i]

        # The ray from the camera to the 2d point light
        o = cam.pos
        x = cam.apply(camera_from_screen_and_depth(uv, jnp.array(1.0), intr))
        v = x - o

        # The mode of the restriction of the Gaussian to the ray
        # from unprojecting the 2d point light at uv.
        t = bilinear(mu - o, v, prec) / bilinear(v, v, prec)

        # Distance to Gaussian w.r.t. inner product norm
        dist = jnp.sqrt(bilinear(o + t * v - mu, o + t * v - mu, prec))

        # Volume of the 3-ball and
        # the length of the intersection of the ray with the 3-ball
        # TODO: Check if this is correct
        vol = 4 / 3 * jnp.pi
        len = 2 * jnp.sqrt(1 - dist**2)

        inlier_logp = jnp.where(dist <= 1.0, jnp.log(len) - jnp.log(vol), -jnp.inf)
        outlier_logp = -jnp.log(intr.width * intr.height)

        in_bounds = (uv[0] <= intr.width) * (uv[1] <= intr.height)

        return jnp.where(
            in_bounds, jnp.where(is_outlier, outlier_logp, inlier_logp), -jnp.inf
        )


mixture_step_hack = MixtureStepHack()


def add_dummy_var(d: ExactDensity):
    """
    Adds a `dummy` variable to a distribution to make it easily mappable while keeping the other args fixed.
    """

    return genjax.exact_density(
        lambda key, dummy, *args: d.sample(key, *args),
        lambda x, dummy, *args: d.logpdf(x, *args),
    )


def random_color_by_cluster(key, gps: DynamicGPS):
    cluster_colors = jax.random.uniform(key, (gps.num_clusters, 3))
    return cluster_colors[gps.cluster_assignments]
