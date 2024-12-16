import genjax
import ipywidgets as widgets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from genjax import Pytree, uniform
from ipywidgets import interact


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

    def __getitem__(self, idx):
        return Uniform(self.min[idx], self.max[idx])


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
        isvalid = jnp.all(max - min >= 2 * epsilon)
        self.epsilon = jnp.where(isvalid, epsilon, jnp.nan * jnp.ones_like(epsilon))
        self.min = jnp.where(isvalid, min, jnp.nan * jnp.ones_like(min))
        self.max = jnp.where(isvalid, max, jnp.nan * jnp.ones_like(max))

    def __getitem__(self, idx):
        return NiceTruncatedCenteredUniform(
            self.epsilon[idx], self.min[idx], self.max[idx]
        )

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
        joint_logpdfs_for_each_uniform = jnp.nan_to_num(
            joint_logpdfs_for_each_uniform, nan=-jnp.inf
        )
        return jax.scipy.special.logsumexp(joint_logpdfs_for_each_uniform)


@Pytree.dataclass
class MixtureOfNiceTruncatedCenteredUniforms(genjax.ExactDensity):
    probs: jnp.ndarray
    ntcus: NiceTruncatedCenteredUniform  # Batched pytree

    def sample(self, key, center):
        k1, k2 = jax.random.split(key)
        keys = jax.random.split(k1, len(self.probs))
        idx = genjax.categorical.sample(k2, jnp.log(self.probs))
        vals = jax.vmap(lambda key: self.ntcus.sample(key, center))(keys)
        return vals[idx]

    def logpdf(self, x, center):
        logpdfs_given_ntcu = jax.vmap(lambda ntcu: ntcu.logpdf(x, center))(self.ntcus)
        joint_logpdfs_for_each_ntcu = jnp.log(self.probs) + logpdfs_given_ntcu
        joint_logpdfs_for_each_ntcu = jnp.nan_to_num(
            joint_logpdfs_for_each_ntcu, nan=-jnp.inf
        )
        return jax.scipy.special.logsumexp(joint_logpdfs_for_each_ntcu)


def prior_and_obsmodel_are_compatible(
    prior: Uniform,
    obsmodel: NiceTruncatedCenteredUniform,
):
    """
    Checks that the assumptions of the posterior functions are met.
    """
    return jnp.logical_and(obsmodel.min <= prior.min, obsmodel.max >= prior.max)


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
    # Also return nans if P(obs) = 0.
    isvalid = assumptions_are_met & (minval <= maxval)
    minval = jnp.where(isvalid, minval, jnp.nan)
    maxval = jnp.where(isvalid, maxval, jnp.nan)

    return Uniform(minval, maxval)


def get_marginal_probability_of_obs_for_uniform_prior_ntcu_obs(
    prior: Uniform, obsmodel: NiceTruncatedCenteredUniform, obs: float
) -> float:
    """
    Given that under P, x ~ prior and y ~ obsmodel(x), this returns the marginal
    probability P(y = obs).
    """
    posterior = get_posterior_for_uniform_prior_ntcu_obs(prior, obsmodel, obs)
    pobs_is_0 = jnp.isnan(posterior.min)
    region_size = posterior.max - posterior.min
    joint_pdf = 1 / (prior.max - prior.min) * 1 / (2 * obsmodel.epsilon)
    return jnp.where(pobs_is_0, 0.0, region_size * joint_pdf)


def get_posterior_from_mix_of_uniform_prior_and_mix_of_nctus_obs(
    prior: MixtureOfUniforms,
    obsmodel: MixtureOfNiceTruncatedCenteredUniforms,
    obs: float,
) -> MixtureOfUniforms:
    """
    Given that under P, x ~ prior and y ~ obsmodel(x), this returns the posterior P(x | y).
    """
    all_ij_pairs = all_pairs(
        jnp.arange(len(prior.probs)), jnp.arange(len(obsmodel.probs))
    )

    # Shape: (len(prior.probs), len(obsmodel.probs))
    p_obs_given_branch = jax.vmap(
        lambda ij: get_marginal_probability_of_obs_for_uniform_prior_ntcu_obs(
            prior.uniforms[ij[0]], obsmodel.ntcus[ij[1]], obs
        )
    )(all_ij_pairs)

    prior_probs_of_branches = jax.vmap(
        lambda ij: prior.probs[ij[0]] * obsmodel.probs[ij[1]]
    )(all_ij_pairs)

    joint_probs_of_branches = prior_probs_of_branches * p_obs_given_branch
    posterior_probs_of_branches = joint_probs_of_branches / jnp.sum(
        joint_probs_of_branches
    )

    uniform_dists_per_branch = jax.vmap(
        lambda ij: get_posterior_for_uniform_prior_ntcu_obs(
            prior.uniforms[ij[0]], obsmodel.ntcus[ij[1]], obs
        )
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


### IPy vizualization ###


def create_interactive_posterior_viz(
    prior,
    obs_model,
    prior_min,
    prior_max,
    obs_min,
    obs_max,
    get_exact_posterior_pdf,  # (obs, latent) -> pdf
    get_enum_posterior_pdf=None,  # (obs, gridpoints) -> approx pdf at each gridpoint
):
    if get_enum_posterior_pdf is None:

        def get_enum_posterior_pdf(obs, gridpoints):
            return _get_enum_posterior_pdf(prior, obs_model, obs, gridpoints)

    # Define the grid for prior and observation
    grid_latent = jnp.linspace(prior_min, prior_max, 1001)
    grid_obs = jnp.linspace(obs_min, obs_max, 1001)

    # Compute the prior densities on the latent grid
    prior_pdf = jnp.exp(jax.vmap(prior.logpdf)(grid_latent))

    # Set up the widgets for latent and observed values
    latent_slider = widgets.FloatSlider(
        min=prior_min, max=prior_max, step=0.01, description="Latent Value"
    )
    obs_slider = widgets.FloatSlider(
        min=obs_min, max=obs_max, step=0.01, description="Observed Value"
    )

    # Function to update the plots
    def update_plots(latent_value, obs_value):
        # Clear the current figure
        plt.figure(figsize=(10, 8))

        # Top half: Prior and observation model
        plt.subplot(2, 1, 1)
        plt.title("Prior and Observation Model")

        # Prior plot with latent value line
        plt.plot(grid_latent, prior_pdf, label="Prior PDF", color="blue")
        plt.axvline(x=latent_value, color="red", linestyle="--", label="Latent Value")
        plt.legend()
        plt.ylabel("Density")

        # Observation model at latent value
        obs_pdf = jnp.exp(
            jax.vmap(lambda obs: obs_model.logpdf(obs, latent_value))(grid_obs)
        )
        plt.plot(grid_obs, obs_pdf, label="Observation PDF", color="green")
        plt.axvline(x=obs_value, color="orange", linestyle="--", label="Observed Value")
        plt.legend()

        # Bottom half: Posterior distribution
        plt.subplot(2, 1, 2)
        plt.title("Posterior Distributions")

        plt.plot(grid_latent, prior_pdf, label="Prior PDF", color="blue")

        # Exact posterior based on the observed value
        exact_posterior_pdf = jax.vmap(
            lambda latent: get_exact_posterior_pdf(obs_value, latent)
        )(grid_latent)
        plt.plot(grid_latent, exact_posterior_pdf, label="Exact Posterior", color="red")
        plt.axvline(x=obs_value, color="orange", linestyle="--", label="Observed Value")

        # Enumeration-based posterior (if provided)
        if get_enum_posterior_pdf:
            tight_grid_latent = jnp.linspace(prior_min, prior_max, 1000001)
            enum_posterior_pdf_tight = get_enum_posterior_pdf(
                obs_value, tight_grid_latent
            )
            plot_enum_posterior_pdf = enum_posterior_pdf_tight[::1000]
            plt.plot(
                grid_latent,
                plot_enum_posterior_pdf,
                label="Enumeration Posterior",
                color="purple",
                linestyle="--",
            )

        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Density")

        plt.tight_layout()
        plt.show()

    interact(update_plots, latent_value=latent_slider, obs_value=obs_slider)


def _get_enum_posterior_pdf(prior, obs_model, obs, gridpoints):
    prior_vals = jnp.exp(jax.vmap(prior.logpdf)(gridpoints))
    obs_vals_given_latent = jnp.exp(
        jax.vmap(obs_model.logpdf, in_axes=(None, 0))(obs, gridpoints)
    )
    joint_pdf = prior_vals * obs_vals_given_latent
    posterior_prob_per_point = joint_pdf / jnp.sum(joint_pdf)
    distance_between_points = gridpoints[1] - gridpoints[0]
    posterior_pdf = posterior_prob_per_point / distance_between_points
    return posterior_pdf
