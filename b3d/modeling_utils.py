import genjax
from b3d.pose import sample_uniform_pose, logpdf_uniform_pose, sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from genjax import Mask

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

### Below are placeholders for genjax functions which are currently buggy ###

# There is currently a bug in `genjax.uniform.logpdf`; this `uniform`
# can be used instead until a fix is pushed.
uniform = genjax.exact_density(
    lambda key, low, high: genjax.uniform.sample(key, low, high),
    lambda x, low, high: jnp.sum(genjax.uniform.logpdf(x, low, high))
)

def tfp_distribution(dist):
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return jnp.sum(d.log_prob(v))

    return genjax.exact_density(sampler, logpdf)

normal = tfp_distribution(tfp.distributions.Normal)

def masked_scan_combinator(step, **scan_kwargs):
    """
    Given a generative function `step` so that `step.scan(n=N)` is valid,
    return a generative function accepting an input
    `(initial_state, masked_input_values_array)` and returning a pair
    `(masked_final_state, masked_returnvalue_sequence)`.
    This operates similarly to `step.scan`, but the input values can be masked.
    """
    mstep = step.mask().dimap(
        pre=lambda masked_state, masked_inval: (
            jnp.logical_and(masked_state.flag, masked_inval.flag),
            masked_state.value,
            masked_inval.value
        ),
        post=lambda args, masked_retval: (
            Mask(masked_retval.flag, masked_retval.value[0]),
            Mask(masked_retval.flag, masked_retval.value[1])
        )
    )

    # This should be given a pair (
    #     Mask(True, initial_state), 
    #     Mask(bools_indicating_active, input_vals)
    # ).
    # It wll output a pair (masked_final_state, masked_returnvalue_sequence).
    scanned = mstep.scan(**scan_kwargs)

    scanned_nice = scanned.dimap(
        pre=lambda initial_state, masked_input_values: (
            Mask(True, initial_state),
            Mask(masked_input_values.flag, masked_input_values.value)
        ),
        post=lambda args, retval: retval
    )

    return scanned_nice

def variable_length_unfold_combinator(step, **scan_kwargs):
    """
    Step should accept one arg, `state`, as input,
    and should return a pair `(new_state, retval_for_this_timestep)`.
    """
    scanned = masked_scan_combinator(step, **scan_kwargs)
    return scanned.dimap(
        pre=lambda initial_state, n_steps: (
            initial_state,
            Mask(jnp.array())
        )
    )