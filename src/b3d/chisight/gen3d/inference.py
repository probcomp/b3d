from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff, Pytree
from genjax import UpdateProblemBuilder as U
from jax.random import split
from tqdm import tqdm

import b3d
from b3d.chisight.gen3d.inference_moves import (
    get_pose_proposal_density,
    propose_other_latents_given_pose,
    propose_pose,
)
from b3d.chisight.gen3d.model import (
    dynamic_object_generative_model,
    get_hypers,
    get_new_state,
    make_colors_choicemap,
    make_depth_nonreturn_prob_choicemap,
    make_visibility_prob_choicemap,
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
    - prev_color_proposal_laplace_scale: Scale parameter for proposing point colors
        around the previous point RGB.
    - obs_color_proposal_laplace_scale: Scale parameter for proposing point colors
        around the observed point RGB.
    """

    n_poses: int = Pytree.static()
    do_stochastic_color_proposals: bool
    pose_proposal_std: float
    pose_proposal_conc: float
    prev_color_proposal_laplace_scale: float
    obs_color_proposal_laplace_scale: float


def get_initial_trace(key, hyperparams, initial_state, initial_observed_rgbd):
    """
    Get the initial trace, given the initial state.
    """
    choicemap = (
        C.d(
            {
                "pose": initial_state["pose"],
                "color_scale": initial_state["color_scale"],
                "depth_scale": initial_state["depth_scale"],
                "rgbd": initial_observed_rgbd,
            }
        )
        ^ make_visibility_prob_choicemap(initial_state["visibility_prob"])
        ^ make_colors_choicemap(initial_state["colors"])
        ^ make_depth_nonreturn_prob_choicemap(initial_state["depth_nonreturn_prob"])
    )
    trace, _ = dynamic_object_generative_model.importance(
        key, choicemap, (hyperparams, initial_state)
    )
    return trace


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
                Diff.unknown_change(get_new_state(trace)),
            ),
            C.kw(rgbd=observed_rgbd),
        ),
    )
    return trace


DEFAULT_C2F_SEQ = [(0.04, 1000.0), (0.02, 1500.0), (0.005, 2000.0)]


def inference_step_c2f(
    key, n_seq, n_poses_per_sequential_step, old_trace, observed_rgbd, *args, **kwargs
):
    """
    Take an inference step using a coarse-to-fine sweep of pose proposals.
    At each step of C2F, we propose `n_seq * n_poses_per_sequential_step` poses,
    for each pose, propose all the other latents, and then resample one among
    these options.  That pose is used as the center of the pose proposal
    distribution for the next step of C2F.
    The final trace is returned.

    Args:
    - key: PRNGKey
    - n_seq: For each step of C2F, how many parallel batches of poses to propose.
        (This is provided so more poses can be considered than can fit into GPU memory.)
    - n_poses_per_sequential_step: How many poses to propose in parallel at each step.
        (So at each step of C2F, we propose n_seq * n_poses_per_sequential_step poses,
        and resample one.  Then at the next step of C2F, we propose that many poses
        again, but with a narrower proposal distribution.)
    - old_trace: The trace from the previous timestep.
    - observed_rgbd: The observed RGBD image at the current timestep.
    - **kwargs: Kwargs providing each field of InferenceHyperparams
        other than `n_poses`, `pose_proposal_std`, and `pose_proposal_conc`.
    """
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
    pose_proposal_std_conc_seq=DEFAULT_C2F_SEQ,
    **inference_hyperparam_kwargs,
):
    for std, conc in pose_proposal_std_conc_seq:
        inference_hyperparams = InferenceHyperparams(
            n_poses=n_poses_per_sequential_step,
            pose_proposal_std=std,
            pose_proposal_conc=conc,
            **inference_hyperparam_kwargs,
        )
        key, _ = split(key)
        trace, _ = infer_latents_using_sequential_proposals(
            key, n_seq, trace, inference_hyperparams
        )

    return trace


def inference_step_using_sequential_proposals(
    key, n_seq, old_trace, observed_rgbd, inference_hyperparams
):
    """
    Like `inference_step`, but does `n_seq` sequential proposals
    of `inference_hyperparams.n_poses` poses and other latents,
    and resamples one among all of these.
    Considers n_seq * inference_hyperparams.n_poses proposals in total.
    Returns `(trace, weight)`.
    """
    k1, k2 = split(key)
    trace = advance_time(k1, old_trace, observed_rgbd)
    return infer_latents_using_sequential_proposals(
        k2, n_seq, trace, inference_hyperparams
    )


def infer_latents_using_sequential_proposals(key, n_seq, trace, inference_hyperparams):
    shared_args = (trace, inference_hyperparams)

    def get_weight(key):
        return infer_latents(key, *shared_args, get_trace=False, get_metadata=False)[0]

    k1, k2 = split(key)
    ks = split(k1, n_seq)
    weights = [get_weight(k) for k in ks]

    normalized_logps = jax.nn.log_softmax(jnp.array(weights))
    chosen_idx = jax.random.categorical(k2, normalized_logps)
    trace, _ = infer_latents(ks[chosen_idx], *shared_args, get_metadata=False)
    overall_weight = jax.scipy.special.logsumexp(jnp.array(weights))

    return trace, overall_weight


def inference_step(
    key, old_trace, observed_rgbd, inference_hyperparams, *args, **kwargs
):
    """
    Perform over the latent state at time T, given the observed
    rgbd at this timestep, and the old trace from time T-1.

    Also returns an estimate of the marginal likelihood of
    the observed rgbd, given the latent state from time T-1.

    All arguments after `inference_hyperparams` are passed to
    `infer_latents`; see `infer_latents` for details.
    """
    k1, k2 = split(key)
    trace = advance_time(k1, old_trace, observed_rgbd)
    return infer_latents(k2, trace, inference_hyperparams, *args, **kwargs)


@partial(jax.jit, static_argnums=(3, 4, 5))
def infer_latents(
    key,
    trace,
    inference_hyperparams,
    get_trace=True,
    get_weight=True,
    get_metadata=True,
    # If this is included, we guarantee that this is one of the
    # poses in the grid.
    use_gt_pose=False,
    gt_pose=b3d.Pose.identity(),
    # Useful for debugging: turn off - logq in the pose resampling
    include_qscores_in_outer_resample=True,
):
    """
    Infer the latents at time `T`, given a trace `T` with arguments
    containing the prev state (state at `T-1`).
    Pose proposals are centered around the trace in the new state
    in this trace.

    Args:
    - key: PRNGKey
    - trace: Partially inferred trace at time `T`.
        (E.g. the output of `advance_time`.)
    - inference_hyperparams: InferenceHyperparams
    - get_trace: Controls whether the inferred trace is in the function's return value.
    - get_weight: Controls whether the weight is in the function's return value.
    - get_metadata: Controls whether the metadata is in the function's return value.
    - use_gt_pose: If true, the value `gt_pose` will be placed as the first
        proposed pose, in the pose proposal.  (Ie. the function will act as though
        it proposed this pose on the first step.)
    - gt_pose: The ground truth pose at time T.
    """
    _, k2, k3, k4 = split(key, 4)

    pose_generation_keys = split(k2, inference_hyperparams.n_poses)
    proposed_poses, log_q_poses = jax.vmap(propose_pose, in_axes=(0, None, None))(
        pose_generation_keys, trace, inference_hyperparams
    )

    proposed_poses = jax.tree.map(
        lambda x, y: x.at[0].set(jnp.where(use_gt_pose, y, x[0])),
        proposed_poses,
        gt_pose,
    )
    log_q_poses = log_q_poses.at[0].set(
        jnp.where(
            use_gt_pose,
            get_pose_proposal_density(gt_pose, trace, inference_hyperparams),
            log_q_poses[0],
        )
    )

    param_generation_keys = split(k3, inference_hyperparams.n_poses)
    proposed_traces, log_q_nonpose_latents, other_latents_metadata = jax.vmap(
        propose_other_latents_given_pose, in_axes=(0, None, 0, None)
    )(param_generation_keys, trace, proposed_poses, inference_hyperparams)
    p_scores = jax.vmap(lambda tr: tr.get_score())(proposed_traces)

    scores = jnp.where(
        include_qscores_in_outer_resample,
        p_scores - log_q_poses - log_q_nonpose_latents,
        p_scores,
    )
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


def run_inference_many_frames(
    key,
    trace,
    all_data,
    inference_hyperparams,
    use_gt_pose=True,
    gt_poses=None,
    get_metadata=False,
    include_qscores_in_outer_resample=True,
):
    traces = []
    if gt_poses is None:
        gt_poses = [b3d.Pose.identity()] * len(all_data)
    for T in tqdm(len(all_data)):
        key = b3d.split_key(key)
        trace, _ = inference_step(
            key,
            trace,
            all_data[T]["rgbd"],
            inference_hyperparams,
            use_gt_pose=use_gt_pose,
            gt_pose=gt_poses[T],
            get_metadata=get_metadata,
            include_qscores_in_outer_resample=include_qscores_in_outer_resample,
        )
        traces.append(trace)
    return trace


### Utils ###


def logmeanexp(vec):
    vec = jnp.where(jnp.isnan(vec), -jnp.inf, vec)
    return jax.scipy.special.logsumexp(vec) - jnp.log(len(vec))
