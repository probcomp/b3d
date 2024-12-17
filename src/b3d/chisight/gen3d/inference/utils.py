import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U


def logmeanexp(vec):
    vec = jnp.where(jnp.isnan(vec), -jnp.inf, vec)
    return jax.scipy.special.logsumexp(vec) - jnp.log(len(vec))


def update_field(key, trace, fieldname, value):
    """
    Update `trace` by changing the value at address `fieldname` to `value`.
    Returns a new trace.
    """
    return update_fields(key, trace, [fieldname], [value])


def update_fields(key, trace, fieldnames, values):
    """
    Update `trace` by changing the values at the addresses in `fieldnames` to the
    corresponding values in `values`.  Returns a new trace.
    """
    hyperparams, previous_state = trace.get_args()
    print("previous_state: ", previous_state)
    jax.debug.print("previous_state: {v}", v=previous_state)
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (Diff.no_change(hyperparams), Diff.no_change(previous_state)),
            C.kw(**dict(zip(fieldnames, values))),
        ),
    )
    return trace


def update_vmapped_fields(key, trace, fieldnames, values):
    """
    For each `fieldname` in fieldnames, and each array `arr` in the
    corresponding slot in `values`, updates `trace` at addresses
    (0, fieldname) through (len(arr) - 1, fieldname) to the corresponding
    values in `arr`.
    (That is, this assumes for each fieldname, there is a vmap combinator
    sampled at that address in the trace.)
    """
    c = C.n()
    for addr, val in zip(fieldnames, values):
        c = c ^ jax.vmap(lambda idx: C[addr, idx].set(val[idx]))(
            jnp.arange(val.shape[0])
        )

    hyperparams, previous_state = trace.get_args()
    trace, _, _, _ = trace.update(
        key,
        U.g((Diff.no_change(hyperparams), Diff.no_change(previous_state)), c),
    )
    return trace


def update_vmapped_field(key, trace, fieldname, value):
    """
    For information, see `update_vmapped_fields`.
    """
    return update_vmapped_fields(key, trace, [fieldname], [value])


def all_pairs(X, Y):
    """
    Return an array `ret` of shape (|X| * |Y|, 2) where each row
    is a pair of values from X and Y.
    That is, `ret[i, :]` is a pair [x, y] for some x in X and y in Y.
    """
    return jnp.swapaxes(jnp.stack(jnp.meshgrid(X, Y), axis=-1), 0, 1).reshape(-1, 2)


def normalize_log_scores(scores):
    """
    Util for constructing log resampling distributions, avoiding NaN issues.

    (Conversely, since there will be no NaNs, this could make it harder to debug.)
    """
    val = scores - jax.scipy.special.logsumexp(scores)
    return jnp.where(
        jnp.any(jnp.isnan(val)), -jnp.log(len(val)) * jnp.ones_like(val), val
    )
