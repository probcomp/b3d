from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnames=("target", "proposal", "S","R"))
def vector_SIR(key, target, target_args, proposal, proposal_args,*, S=1, R=1):
    """
    Args:
        target: Callable that returns a log score for vector of proposals `xs`; Signature (key, xs, *target_args)
        target_args: Additional arguments to `target`
        proposal: Callable that returns a vector of weights `ws` and proposals `xs`; signature (key, *proposal_args)
        proposal_args: Additional arguments to `proposal`
        S: Number of Samples
        R: Number of Re-Samples

    Returns:
        Re-sampled vectors and weights
    """
    key = jax.random.split(key)[1]
    keys = jax.random.split(key, S)
    proposal_weights, proposed_samples = jax.vmap(proposal, (0,) + (None,)*len(proposal_args))(keys, *proposal_args)
    N = proposed_samples.shape[1]

    key = jax.random.split(key)[1]
    keys = jax.random.split(key, S)
    target_weights = jax.vmap(target, (0,0) + (None,)*len(target_args))(keys, proposed_samples, *target_args)

    ws = target_weights - proposal_weights
    ws = ws  - logsumexp(ws)

    # Resample
    key = jax.random.split(key)[1]
    winners = jax.random.categorical(key, ws, axis=0, shape=(R,N))

    xs_ = proposed_samples[winners, jnp.arange(N)]
    ws_ = ws[winners, jnp.arange(N)]

    return xs_, ws_
