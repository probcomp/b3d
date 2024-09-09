import genjax
import jax
import jax.numpy as jnp
from genjax import Pytree
from tensorflow_probability.substrates import jax as tfp

from b3d.pose import (
    logpdf_gaussian_vmf_pose,
    logpdf_uniform_pose,
    sample_gaussian_vmf_pose,
    sample_uniform_pose,
)


@jax.jit
def sample_uniform_broadcasted(key, low, high):
    return genjax.uniform.sample(key, low, high)


def logpdf_uniform_broadcasted(values, low, high):
    valid = (low <= values) & (values <= high)
    position_score = jnp.log((valid * 1.0) * (jnp.ones_like(values) / (high - low)))
    return position_score.sum()


uniform_broadcasted = genjax.exact_density(
    sample_uniform_broadcasted, logpdf_uniform_broadcasted
)

uniform_discrete = genjax.exact_density(
    lambda key, vals: jax.random.choice(key, vals),
    lambda sampled_val, vals: jnp.log(1.0 / (vals.shape[0])),
)
uniform_pose = genjax.exact_density(sample_uniform_pose, logpdf_uniform_pose)

vmf = genjax.exact_density(
    lambda key, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).sample(seed=key),
    lambda x, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).log_prob(x),
)

gaussian_vmf = genjax.exact_density(sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose)

### Below are placeholders for genjax functions which are currently buggy ###
## TODO: these bugs in genjax should now be fixed, so we should be able to
# remove these.

# There is currently a bug in `genjax.uniform.logpdf`; this `uniform`
# can be used instead until a fix is pushed.
uniform = genjax.exact_density(
    lambda key, low, high: genjax.uniform.sample(key, low, high),
    lambda x, low, high: jnp.sum(genjax.uniform.logpdf(x, low, high)),
)


def tfp_distribution(dist):
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return jnp.sum(d.log_prob(v))

    return genjax.exact_density(sampler, logpdf)


categorical = tfp_distribution(
    lambda logits: tfp.distributions.Categorical(logits=logits)
)
bernoulli = tfp_distribution(lambda logits: tfp.distributions.Bernoulli(logits=logits))
normal = tfp_distribution(tfp.distributions.Normal)

### Mixture distribution combinator ###


@Pytree.dataclass
class PythonMixtureDistribution(genjax.ExactDensity):
    """
    Mixture of different distributions.
    Constructor:
    - dists : python list of N genjax.ExactDensity objects

    Distribution args:
    - probs : (N,) array of branch probabilities
    - args : python list of argument tuples, so that
        `dists[i].sample(key, *args[i])` is valid for each i
    """

    dists: any = genjax.Pytree.static()

    def sample(self, key, probs, args):
        values = []
        for i, dist in enumerate(self.dists):
            key, subkey = jax.random.split(key)
            values.append(dist.sample(subkey, *args[i]))
        values = jnp.array(values)
        key, subkey = jax.random.split(key)
        component = genjax.categorical.sample(subkey, jnp.log(probs))
        return values[component]

    def logpdf(self, observed, probs, args):
        logprobs = []
        for i, dist in enumerate(self.dists):
            lp = dist.logpdf(observed, *args[i])
            logprobs.append(lp + jnp.log(probs[i]))
        logprobs = jnp.stack(logprobs)
        return jax.scipy.special.logsumexp(logprobs)


### Truncated laplace distribution, and mapped version for RGB ###


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
        assert low < high
        assert low + uniform_window_size < high - uniform_window_size
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
        laplace_logp_below_low = tfp.distributions.Laplace(loc, scale).log_cdf(low)
        laplace_logp_above_high = tfp.distributions.Laplace(
            loc, scale
        ).log_survival_function(high)
        log_window_size = jnp.log(uniform_window_size)

        return jnp.where(
            jnp.logical_and(
                low + uniform_window_size < obs, obs < high - uniform_window_size
            ),
            laplace_logpdf,
            jnp.where(
                obs < low + uniform_window_size,
                jnp.logaddexp(laplace_logp_below_low - log_window_size, laplace_logpdf),
                jnp.logaddexp(
                    laplace_logp_above_high - log_window_size, laplace_logpdf
                ),
            ),
        )


truncated_laplace = TruncatedLaplace()

_FIXED_COLOR_UNIFORM_WINDOW = 1 / 255


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
