from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff, Pytree
from genjax import UpdateProblemBuilder as U
from jax.random import split

from b3d.chisight.gen3d.inference_moves import (
    get_pose_proposal_density,
    propose_other_latents_given_pose,
    propose_pose,
)
from b3d.chisight.gen3d.model import (
    get_hypers,
    get_prev_state,
)


@Pytree.dataclass
class InferenceHyperparams(Pytree):
    """
    Parameters for the inference algorithm.
    - n_poses: Number of poses to propose at each timestep.
    - do_stochastic_color_proposals: If true, the color proposal will be
        absolutely continuous w.r.t. the Lebesgue measure on [0, 1]^3.
        If false, the color proposal will consider returning exactly the
        old color, and exactly the new color.
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

    n_poses: int = Pytree.static()
    do_stochastic_color_proposals: bool = Pytree.static()
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


DEFAULT_C2F_SEQ = [(0.04, 1000.0), (0.02, 1500.0), (0.005, 2000.0)]


def inference_step_c2f(
    key, n_seq, n_poses_per_sequential_step, old_trace, observed_rgbd, *args, **kwargs
):
    k1, k2 = split(key)
    trace = advance_time(k1, old_trace, observed_rgbd)
    return infer_latents_c2f(
        k2, n_seq, n_poses_per_sequential_step, trace, *args, **kwargs
    )


def infer_latents_c2f(
    key,
    n_seq,
    n_poses_per_sequential_step,
    trace,
    effective_color_transition_scale,
    do_stochastic_color_proposals=True,
    pose_proposal_std_conc_seq=DEFAULT_C2F_SEQ,
):
    for std, conc in pose_proposal_std_conc_seq:
        inference_hyperparams = InferenceHyperparams(
            n_poses=n_poses_per_sequential_step,
            do_stochastic_color_proposals=do_stochastic_color_proposals,
            pose_proposal_std=std,
            pose_proposal_conc=conc,
            effective_color_transition_scale=effective_color_transition_scale,
        )
        key, _ = split(key)
        trace, _ = infer_latents_using_sequential_proposals(
            key, n_seq, trace, inference_hyperparams
        )

    return trace


def inference_step_using_sequential_proposals(
    key, n_seq, old_trace, observed_rgbd, inference_hyperparams
):
    k1, k2 = split(key)
    trace = advance_time(k1, old_trace, observed_rgbd)
    return infer_latents_using_sequential_proposals(
        k2, n_seq, trace, inference_hyperparams
    )


def infer_latents_using_sequential_proposals(key, n_seq, trace, inference_hyperparams):
    """
    Like `inference_step`, but does `n_seq` sequential proposals
    of `inference_hyperparams.n_poses` poses and other latents,
    and resamples one among all of these.
    Returns `(trace, weight)`.
    """
    shared_args = (trace, inference_hyperparams)

    def get_weight(key):
        return infer_latents(key, *shared_args, get_trace=False, get_metadata=False)[0]

    k1, k2 = split(key)
    ks = split(k1, n_seq)
    weights = [get_weight(k) for k in ks]
    print("weights: ", weights)

    normalized_logps = jax.nn.log_softmax(jnp.array(weights))
    chosen_idx = jax.random.categorical(k2, normalized_logps)
    trace, _ = infer_latents(ks[chosen_idx], *shared_args, get_metadata=False)
    overall_weight = jax.scipy.special.logsumexp(jnp.array(weights))

    return trace, overall_weight


def inference_step(
    key, old_trace, observed_rgbd, inference_hyperparams, *args, **kwargs
):
    k1, k2 = split(key)
    trace = advance_time(k1, old_trace, observed_rgbd)
    return infer_latents(k2, trace, inference_hyperparams, *args, **kwargs)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def infer_latents(
    key,
    trace,
    inference_hyperparams,
    get_trace=True,
    get_weight=True,
    get_metadata=True,
    # If this is included, we guarantee that this is one of the
    # poses in the grid.
    gt_pose=None,
):
    """
    Perform over the latent state at time T, given the observed
    rgbd at this timestep, and the old trace from time T-1.

    Also returns an estimate of the marginal likelihood of
    the observed rgbd, given the latent state from time T-1.

    If `gt_pose` is not None, this will force the pose sampled at index 0
    in the sampling step to be `gt_pose`.  (That is, this will do inference
    as would occur given that the ground truth pose was the first sampled
    pose.)
    """
    _, k2, k3, _ = split(key, 4)

    pose_generation_keys = split(k2, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None))(
        pose_generation_keys, trace, inference_hyperparams
    )

    if gt_pose is not None:
        proposed_poses = jax.tree.map(
            lambda x, y: x.at[0].set(y), proposed_poses, gt_pose
        )
        log_q_poses = log_q_poses.at[0].set(
            get_pose_proposal_density(gt_pose, trace, inference_hyperparams)
        )

    param_generation_keys = split(k3, inference_hyperparams.n_poses)
    proposed_traces, log_q_nonpose_latents, other_latents_metadata = jax.vmap(
        propose_other_latents_given_pose, in_axes=(0, None, 0, None)
    )(param_generation_keys, trace, proposed_poses, inference_hyperparams)
    p_scores = jax.vmap(lambda tr: tr.get_score())(proposed_traces)

    scores = p_scores  #  - log_q_poses - log_q_nonpose_latents
    chosen_index = jnp.argmax(scores)  # jax.random.categorical(k4, scores)
    new_trace = jax.tree.map(lambda x: x[chosen_index], proposed_traces)

    weight = jnp.max(scores)  # logmeanexp(scores)
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
