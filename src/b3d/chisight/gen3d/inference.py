from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

from .inference_moves import (
    propose_other_latents_given_pose,
    propose_pose,
)
from .model import (
    get_hypers,
    get_prev_state,
)

# Use namedtuple rather than dict so we can hash this, and use it as a static arg to a jitted function.
InferenceHyperparams = namedtuple(
    "InferenceHyperparams", ["n_poses", "pose_proposal_std", "pose_proposal_conc"]
)


@jax.jit
def advance_time(key, trace, observed_rgbd):
    """
    Advance to the next timestep, setting the new latent state to the
    same thing as the previous latent state, and setting the new
    observed RGBD value.

    Returns a trace where previous_state (stored in the arguments)
    and new_state (sampled in the choices and returned) are identical.
    """
    trace, _, _, _ = trace.update(
        key,
        U.g(
            (
                Diff.no_change(get_hypers(trace)),
                Diff.unknown_change(get_prev_state(trace)),
            ),
            C.kw(rgbd=observed_rgbd),
        ),
    )
    return trace


@partial(jax.jit, static_argnums=(3,))
def inference_step(key, old_trace, observed_rgbd, inference_hyperparams):
    """
    Perform over the latent state at time T, given the observed
    rgbd at this timestep, and the old trace from time T-1.

    Also returns an estimate of the marginal likelihood of
    the observed rgbd, given the latent state from time T-1.
    """
    k1, k2, k3, k4 = split(key, 4)

    trace = advance_time(k1, old_trace, observed_rgbd)

    pose_generation_keys = split(k2, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None))(
        pose_generation_keys, trace, inference_hyperparams
    )

    param_generation_keys = split(k3, inference_hyperparams.n_poses)
    proposed_traces, log_q_nonpose_latents, other_latents_metadata = jax.vmap(
        propose_other_latents_given_pose, in_axes=(0, None, 0, None)
    )(param_generation_keys, trace, proposed_poses, inference_hyperparams)
    p_scores = jax.vmap(lambda tr: tr.get_score())(proposed_traces)

    scores = p_scores - log_q_poses - log_q_nonpose_latents
    chosen_index = jax.random.categorical(k4, scores)
    new_trace = jax.tree.map(lambda x: x[chosen_index], proposed_traces)

    return (
        new_trace,
        logmeanexp(scores),
        {
            "proposed_poses": proposed_poses,
            "chosen_pose_index": chosen_index,
            "p_scores": p_scores,
            "log_q_poses": log_q_poses,
            "log_q_nonpose_latents": log_q_nonpose_latents,
            "other_latents_metadata": other_latents_metadata,
        },
    )


def inference_step_noweight(*args):
    """
    Same as inference_step, but only returns the new trace
    (not the weight).
    """
    return inference_step(*args)[0]


### Utils ###


def logmeanexp(vec):
    vec = jnp.where(jnp.isnan(vec), -jnp.inf, vec)
    return jax.scipy.special.logsumexp(vec) - jnp.log(len(vec))
