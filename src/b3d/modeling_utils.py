import genjax
<<<<<<< HEAD:b3d/modeling_utils.py
from b3d.pose import (
    sample_uniform_pose,
    logpdf_uniform_pose,
    sample_gaussian_vmf_pose,
    logpdf_gaussian_vmf_pose,
)
=======
>>>>>>> main:src/b3d/modeling_utils.py
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from b3d.pose import (
    logpdf_gaussian_vmf_pose,
    logpdf_uniform_pose,
    sample_gaussian_vmf_pose,
    sample_uniform_pose,
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
