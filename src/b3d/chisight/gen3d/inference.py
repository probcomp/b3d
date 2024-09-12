from functools import partial, wraps
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

from b3d.chisight.gen3d.inference_moves import (
    propose_other_latents_given_pose,
    propose_pose,
)
from b3d.chisight.gen3d.model import (
    get_hypers,
    get_prev_state,
)


# Use namedtuple rather than dict so we can hash this, and use it as a static arg to a jitted function.
class InferenceHyperparams(NamedTuple):
    """
    Parameters for the inference algorithm.
    - n_poses: Number of poses to propose at each timestep.
    - pose_proposal_std: Standard deviation of the position distribution for the pose.
    - pose_proposal_conc: Concentration parameter for the orientation distribution for the pose.
    - effective_color_transition_scale: This parameter is used in the color proposal.
        When the color transition kernel is a laplace, this should be its scale.
        When the color transition kernel is a different distribution, set this to something
        that would make a laplace transition kernel propose with a somewhat similar spread
        to the kernel you are using.  (This parameter is used to decide
        the size of the proposal in the color proposal, using a simple analysis
        we conducted in the laplace case.)
    """

    n_poses: int
    pose_proposal_std: float
    pose_proposal_conc: float
    effective_color_transition_scale: float


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


def inference_step_using_sequential_proposals(
    key, n_seq, old_trace, observed_rgbd, inference_hyperparams
):
    """
    Like `inference_step`, but does `n_seq` sequential proposals
    of `inference_hyperparams.n_poses` poses and other latents,
    and resamples one among all of these.
    Returns `(trace, weight)`.
    """
    shared_args = (old_trace, observed_rgbd, inference_hyperparams)

    def get_weight(key):
        return inference_step(key, *shared_args, get_trace=False, get_metadata=False)[0]

    k1, k2 = split(key)
    ks = split(k1, n_seq)
    weights = []
    for k in ks:
        weights.append(get_weight(k))

    normalized_logps = jax.nn.log_softmax(jnp.array(weights))
    chosen_idx = jax.random.categorical(k2, normalized_logps)
    trace, _ = inference_step(ks[chosen_idx], *shared_args, get_metadata=False)
    overall_weight = jax.scipy.special.logsumexp(jnp.array(weights))

    return trace, overall_weight


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def inference_step(
    key,
    old_trace,
    observed_rgbd,
    inference_hyperparams,
    get_trace=True,
    get_weight=True,
    get_metadata=True,
):
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

    weight = logmeanexp(scores)
    metadata = {
        "proposed_poses": proposed_poses,
        "chosen_pose_index": chosen_index,
        "p_scores": p_scores,
        "log_q_poses": log_q_poses,
        "log_q_nonpose_latents": log_q_nonpose_latents,
        "other_latents_metadata": other_latents_metadata,
    }

    ret = ()
    if get_trace:
        ret = (*ret, new_trace)
    if get_weight:
        ret = (*ret, weight)
    if get_metadata:
        ret = (*ret, metadata)

    return ret


@wraps(inference_step)
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
