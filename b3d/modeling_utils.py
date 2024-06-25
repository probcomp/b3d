import genjax
from b3d.pose import sample_uniform_pose, logpdf_uniform_pose, sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

uniform_discrete = genjax.exact_density(
    lambda key, vals: jax.random.choice(key, vals),
    lambda sampled_val, vals: jnp.log(1.0 / (vals.shape[0])),
)
uniform_pose = genjax.exact_density(sample_uniform_pose, logpdf_uniform_pose)

vmf = genjax.exact_density(
    lambda key, mean, concentration: tfp.distributions.VonMisesFisher(mean, concentration).sample(seed=key),
    lambda x, mean, concentration: tfp.distributions.VonMisesFisher(mean, concentration).log_prob(x),
)

gaussian_vmf = genjax.exact_density(sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose)