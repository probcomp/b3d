import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree, uniform


@Pytree.dataclass
class Uniform(genjax.ExactDensity):
    """
    A uniform distribution on the interval [min, max].
    """

    min: float
    max: float

    def sample(self, key):
        return uniform.sample(key, self.min, self.max)

    def logpdf(self, x):
        return jnp.where(
            (self.min <= x) & (x <= self.max),
            jnp.log(1 / (self.max - self.min)),
            -jnp.inf,
        )


@Pytree.dataclass
class NiceTruncatedCenteredUniform(genjax.ExactDensity):
    """
    This distribution, given a value `center`, "tries" to sample
    uniformly from the interval [center - epsilon, center + epsilon],
    without going outside the interval [min, max].

    Specifically, the distribution is:
    - If center <= min + epsilon, uniform from [min, min + 2 * epsilon]
    - If center >= max - epsilon, uniform from [max - 2 * epsilon, max]
    - Otherwise, uniform from [center - epsilon, center + epsilon]

    A nice feature of this distribution is that its PDF is always
    either 0 or 1 / (2 * epsilon).
    """

    epsilon: float
    min: float
    max: float

    def __init__(self, epsilon, min, max):
        assert max - min >= 2 * epsilon
        self.epsilon = epsilon
        self.min = min
        self.max = max

    def _get_uniform_given_center(self, center):
        is_near_bottom = center < self.min + self.epsilon
        is_near_top = center > self.max - self.epsilon
        minval = jnp.where(
            is_near_bottom,
            self.min,
            jnp.where(is_near_top, self.max - 2 * self.epsilon, center - self.epsilon),
        )
        maxval = jnp.where(
            is_near_top,
            self.max,
            jnp.where(
                is_near_bottom, self.min + 2 * self.epsilon, center + self.epsilon
            ),
        )
        return Uniform(minval, maxval)

    def sample(self, key, center):
        return self._get_uniform_given_center(center).sample(key)

    def logpdf(self, x, center):
        return self._get_uniform_given_center(center).logpdf(x)


@Pytree.dataclass
class MixtureOfUniforms(genjax.ExactDensity):
    probs: jnp.ndarray
    uniforms: Uniform  # Batched pytree

    def sample(self, key):
        k1, k2 = jax.random.split(key)
        keys = jax.random.split(k1, len(self.probs))
        idx = genjax.categorical.sample(k2, jnp.log(self.probs))
        vals = jax.vmap(lambda key, uniform: uniform.sample(key))(keys, self.uniforms)
        return vals[idx]

    def logpdf(self, x):
        logpdfs_given_uniform = jax.vmap(lambda uniform: uniform.logpdf(x))(
            self.uniforms
        )
        joint_logpdfs_for_each_uniform = jnp.log(self.probs) + logpdfs_given_uniform
        return jax.scipy.special.logsumexp(joint_logpdfs_for_each_uniform)


@Pytree.dataclass
class MixtureOfNiceTruncatedCenteredUniforms(genjax.ExactDensity):
    probs: jnp.ndarray
    ntcus: NiceTruncatedCenteredUniform

    def sample(self, key, center):
        k1, k2 = jax.random.split(key)
        keys = jax.random.split(k1, len(self.probs))
        idx = genjax.categorical.sample(k2, jnp.log(self.probs))
        vals = jax.vmap(lambda key: self.ntcus.sample(key, center))(keys)
        return vals[idx]

    def logpdf(self, x, center):
        logpdfs_given_ntcu = jax.vmap(lambda ntcu: ntcu.logpdf(x, center))(self.ntcus)
        joint_logpdfs_for_each_ntcu = jnp.log(self.probs) + logpdfs_given_ntcu
        return jax.scipy.special.logsumexp(joint_logpdfs_for_each_ntcu)


def prior_and_obsmodel_are_compatible(
    prior: Uniform,
    obsmodel: NiceTruncatedCenteredUniform,
):
    """
    Checks that the assumptions of the posterior functions are met.
    """
    return obsmodel.min <= prior.min and obsmodel.max >= prior.max


def get_posterior_for_uniform_prior_ntcu_obs(
    prior: Uniform, obsmodel: NiceTruncatedCenteredUniform, obs: float
) -> Uniform:
    """
    Given that under P, x ~ prior and y ~ obsmodel(x), this returns the posterior P(x | y).
    """
    assumptions_are_met = prior_and_obsmodel_are_compatible(prior, obsmodel)
    minval = jnp.maximum(prior.min, obs - obsmodel.epsilon)
    maxval = jnp.minimum(prior.max, obs + obsmodel.epsilon)

    # If the assumptions under which this posterior was derived are not
    # met, return a distribution with NaNs to warn the user.
    minval = jnp.where(assumptions_are_met, minval, jnp.nan)
    maxval = jnp.where(assumptions_are_met, maxval, jnp.nan)

    return Uniform(minval, maxval)


def get_marginal_probability_of_obs_for_uniform_prior_ntcu_obs(
    prior: Uniform, obsmodel: NiceTruncatedCenteredUniform, obs: float
) -> float:
    """
    Given that under P, x ~ prior and y ~ obsmodel(x), this returns the marginal
    probability P(y = obs).
    """
    posterior = get_posterior_for_uniform_prior_ntcu_obs(prior, obsmodel, obs)
    region_size = posterior.max - posterior.min
    joint_pdf = 1 / (prior.max - prior.min) * 1 / (2 * obsmodel.epsilon)
    return region_size * joint_pdf


def get_posterior_from_mix_of_uniform_prior_and_mix_of_nctus_obs(
    prior: MixtureOfUniforms,
    obsmodel: MixtureOfNiceTruncatedCenteredUniforms,
    obs: float,
) -> MixtureOfUniforms:
    """
    Given that under P, x ~ prior and y ~ obsmodel(x), this returns the posterior P(x | y).
    """
    all_ij_pairs = all_pairs(len(prior.probs), len(obsmodel.probs))

    # Shape: (len(prior.probs), len(obsmodel.probs))
    p_obs_given_branch = jax.vmap(
        lambda i, j: get_marginal_probability_of_obs_for_uniform_prior_ntcu_obs(
            prior.uniforms[i], obsmodel.ntcus[j], obs
        ),
        in_axes=(0, 0),
    )(all_ij_pairs)

    prior_probs_of_branches = jax.vmap(
        lambda i, j: prior.probs[i] * obsmodel.probs[j], in_axes=(0, 0)
    )(all_ij_pairs)

    joint_probs_of_branches = prior_probs_of_branches * p_obs_given_branch
    posterior_probs_of_branches = joint_probs_of_branches / jnp.sum(
        joint_probs_of_branches
    )

    uniform_dists_per_branch = jax.vmap(
        lambda i, j: get_posterior_for_uniform_prior_ntcu_obs(
            prior.uniforms[i], obsmodel.ntcus[j], obs
        ),
        in_axes=(0, 0),
    )(all_ij_pairs)

    return MixtureOfUniforms(posterior_probs_of_branches, uniform_dists_per_branch)


### Util ###
def all_pairs(X, Y):
    """
    Return an array `ret` of shape (|X| * |Y|, 2) where each row
    is a pair of values from X and Y.
    That is, `ret[i, :]` is a pair [x, y] for some x in X and y in Y.
    """
    return jnp.swapaxes(jnp.stack(jnp.meshgrid(X, Y), axis=-1), 0, 1).reshape(-1, 2)
