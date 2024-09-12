import jax
import jax.numpy as jnp
import jax.random
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax import UpdateProblemBuilder as U
from jax.random import split

from b3d import Pose
from b3d.modeling_utils import uniform

from .model import (
    get_hypers,
    get_n_vertices,
    get_new_state,
    get_observed_rgbd,
    get_prev_state,
)
from .projection import PixelsPointsAssociation


def normalize_log_scores(scores):
    """
    Util for constructing log resampling distributions, avoiding NaN issues.

    (Conversely, since there will be no NaNs, this could make it harder to debug.)
    """
    val = scores - jax.scipy.special.logsumexp(scores)
    return jnp.where(
        jnp.any(jnp.isnan(val)), -jnp.log(len(val)) * jnp.ones_like(val), val
    )


def propose_pose(key, advanced_trace, inference_hyperparams):
    """
    Propose a random pose near the previous timestep's pose.
    Returns (proposed_pose, log_proposal_density).
    """
    previous_pose = get_prev_state(advanced_trace)["pose"]
    ih = inference_hyperparams
    pose = Pose.sample_gaussian_vmf_pose(
        key, previous_pose, ih.pose_proposal_std, ih.pose_proposal_conc
    )
    log_q = Pose.logpdf_gaussian_vmf_pose(
        pose, previous_pose, ih.pose_proposal_std, ih.pose_proposal_conc
    )
    return pose, log_q


def propose_other_latents_given_pose(key, advanced_trace, pose, inference_hyperparams):
    """
    Proposes all latents other than the pose, conditional upon the pose and observed RGBD
    in `advanced_trace`.
    Returns (proposed_trace, log_q) where `propose_trace` is the new trace with the
    proposed latents (and the same pose and observed rgbd as in the given trace).
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2, k3, k4 = split(key, 4)

    trace = update_field(k1, advanced_trace, "pose", pose)

    k2a, k2b = split(k2)
    (
        colors,
        visibility_probs,
        depth_nonreturn_probs,
        log_q_point_attributes,
        point_proposal_metadata,
    ) = propose_all_pointlevel_attributes(k2a, trace, inference_hyperparams)
    trace = update_vmapped_fields(
        k2b,
        trace,
        ["colors", "visibility_prob", "depth_nonreturn_prob"],
        [colors, visibility_probs, depth_nonreturn_probs],
    )
    # TODO: debug these scores -- right now they are causing bad behavior
    # log_q_point_attributes = 0.0

    k3a, k3b = split(k3)
    depth_scale, log_q_ds = propose_depth_scale(k3a, trace)
    trace = update_field(k3b, trace, "depth_scale", depth_scale)

    k4a, k4b = split(k4)
    color_scale, log_q_cs = propose_color_scale(k4a, trace)
    trace = update_field(k4b, trace, "color_scale", color_scale)

    log_q = log_q_point_attributes + log_q_ds + log_q_cs
    return (
        trace,
        log_q,
        {
            "point_attribute_proposal_metadata": point_proposal_metadata,
            "log_q_point_attributes": log_q_point_attributes,
        },
    )


def propose_all_pointlevel_attributes(key, trace, inference_hyperparams):
    """
    Propose a new color, visibility probability, and depth non-return probability
    for every vertex, conditioned upon the other values in `trace`.

    Returns (colors, visibility_probs, depth_nonreturn_probs, log_q, metadata),
    where colors has shape (n_vertices, 3), visibility_probs and depth_nonreturn_probs
    have shape (n_vertices,), log_q (a float) is (an estimate of)
    the overall log proposal density, and metadata is a dict.
    """
    observed_rgbds_per_point = PixelsPointsAssociation.from_hyperparams_and_pose(
        get_hypers(trace), get_new_state(trace)["pose"]
    ).get_point_rgbds(get_observed_rgbd(trace))

    colors, visibility_probs, depth_nonreturn_probs, log_qs, metadata = jax.vmap(
        propose_a_points_attributes, in_axes=(0, 0, 0, None, None, None, None)
    )(
        split(key, get_n_vertices(trace)),
        jnp.arange(get_n_vertices(trace)),
        observed_rgbds_per_point,
        get_prev_state(trace),
        get_new_state(trace),
        get_hypers(trace),
        inference_hyperparams,
    )

    return colors, visibility_probs, depth_nonreturn_probs, log_qs.sum(), metadata


def propose_a_points_attributes(
    key,
    vertex_index,
    observed_rgbd_for_point,
    prev_state,
    new_state,
    hyperparams,
    inference_hyperparams,
):
    """
    Propose a new color, visibility probability, and depth non-return probability
    for the vertex with index `vertex_index`.

    Returns (color, visibility_prob, depth_nonreturn_prob, log_q, metadata),
    where color is a 3-array, visibility_prob and depth_nonreturn_prob are floats,
    log_q (a float) is (a fair estimate of) the log proposal density,
    and metadata is a dict.
    """
    return _propose_a_points_attributes(
        key,
        observed_rgbd_for_point=observed_rgbd_for_point,
        latent_depth=new_state["pose"].apply(hyperparams["vertices"][vertex_index])[2],
        previous_color=prev_state["colors"][vertex_index],
        previous_visibility_prob=prev_state["visibility_prob"][vertex_index],
        previous_dnrp=prev_state["depth_nonreturn_prob"][vertex_index],
        dnrp_transition_kernel=hyperparams["depth_nonreturn_prob_kernel"],
        visibility_transition_kernel=hyperparams["visibility_prob_kernel"],
        color_kernel=hyperparams["color_kernel"],
        obs_rgbd_kernel=hyperparams["image_kernel"].get_rgbd_vertex_kernel(),
        color_scale=new_state["color_scale"],
        depth_scale=new_state["depth_scale"],
        inference_hyperparams=inference_hyperparams,
    )


def _propose_a_points_attributes(
    key,
    observed_rgbd_for_point,
    latent_depth,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    dnrp_transition_kernel,
    visibility_transition_kernel,
    color_kernel,
    obs_rgbd_kernel,
    color_scale,
    depth_scale,
    inference_hyperparams,
    return_metadata=True,
):
    k1, k2 = split(key, 2)

    def score_attribute_assignment(color, visprob, dnrprob):
        visprob_transition_score = visibility_transition_kernel.logpdf(
            visprob, previous_visibility_prob
        )
        dnrprob_transition_score = dnrp_transition_kernel.logpdf(dnrprob, previous_dnrp)
        color_transition_score = color_kernel.logpdf(color, previous_color)
        likelihood_score = obs_rgbd_kernel.logpdf(
            observed_rgbd=observed_rgbd_for_point,
            latent_rgbd=jnp.append(color, latent_depth),
            color_scale=color_scale,
            depth_scale=depth_scale,
            visibility_prob=visprob,
            depth_nonreturn_prob=dnrprob,
        )
        return (
            visprob_transition_score
            + dnrprob_transition_score
            + color_transition_score
            + likelihood_score
        )

    # Say there are V values in visibility_transition_kernel.support
    # and D values in dnrp_transition_kernel.support.

    # (D*V, 2) array of all pairs of values in the support of the two kernels.
    all_visprob_dnrprob_pairs = all_pairs(
        visibility_transition_kernel.support, dnrp_transition_kernel.support
    )

    # Propose a color for each visprob-dnrprob pair.
    rgbs, log_qs_rgb, rgb_proposal_metadata = jax.vmap(
        lambda k, visprob_dnrprob_pair: propose_vertex_color_given_other_attributes(
            key=k,
            visprob=visprob_dnrprob_pair[0],
            dnrprob=visprob_dnrprob_pair[1],
            observed_rgb=observed_rgbd_for_point[:3],
            score_attribute_assignment=score_attribute_assignment,
            previous_rgb=previous_color,
            color_scale=color_scale,
            inference_hyperparams=inference_hyperparams,
            color_kernel=color_kernel,
        )
    )(split(k1, len(all_visprob_dnrprob_pairs)), all_visprob_dnrprob_pairs)

    log_pscores = jax.vmap(
        lambda visprob_dnrprob_pair, rgb: score_attribute_assignment(
            rgb, visprob_dnrprob_pair[0], visprob_dnrprob_pair[1]
        ),
        in_axes=(0, 0),
    )(all_visprob_dnrprob_pairs, rgbs)

    log_weights = log_pscores - log_qs_rgb
    log_normalized_scores = normalize_log_scores(log_weights)
    index = jax.random.categorical(k2, log_normalized_scores)

    rgb = rgbs[index]
    visibility_prob, dnr_prob = all_visprob_dnrprob_pairs[index]
    log_q_score = log_normalized_scores[index] + log_qs_rgb[index]

    return (
        rgb,
        visibility_prob,
        dnr_prob,
        log_q_score,
        {
            "log_qs_rgb": log_qs_rgb,
            "log_normalized_scores": log_normalized_scores,
            "index": index,
            "all_visprob_dnrprob_pairs": all_visprob_dnrprob_pairs,
            "rgb_proposal_metadata": rgb_proposal_metadata,
        },
    )


def propose_vertex_color_given_other_attributes(
    key,
    visprob,
    dnrprob,
    observed_rgb,
    score_attribute_assignment,
    previous_rgb,
    color_scale,
    inference_hyperparams,
    color_kernel,
):
    value_if_observed_is_valid, log_q_if_valid, metadata_if_valid = (
        propose_vertex_color_given_other_attributes_for_valid_observed_rgb(
            key,
            visprob,
            dnrprob,
            observed_rgb,
            score_attribute_assignment,
            previous_rgb,
            color_scale,
            inference_hyperparams,
        )
    )
    value_if_observed_is_invalid = color_kernel.sample(key, previous_rgb)
    log_q_if_invalid = color_kernel.logpdf(value_if_observed_is_invalid, previous_rgb)

    isvalid = ~jnp.any(observed_rgb < 0)
    value = jnp.where(isvalid, value_if_observed_is_valid, value_if_observed_is_invalid)
    log_q = jnp.where(isvalid, log_q_if_valid, log_q_if_invalid)
    metadata = {
        "isvalid": isvalid,
        "metadata_if_valid": metadata_if_valid,
        "value_if_observed_is_invalid": value_if_observed_is_invalid,
        "log_q_if_invalid": log_q_if_invalid,
    }
    return value, log_q, metadata


def propose_vertex_color_given_other_attributes_for_valid_observed_rgb(
    key,
    visprob,
    dnrprob,
    observed_rgb,
    score_attribute_assignment,
    previous_rgb,
    color_scale,
    inference_hyperparams,
):
    """
    This samples an rgb value from a proposal which first proposes 3 different rgb values,
    then resamples one.  The log q score is estimated using
    simple logic for "filling in the auxiliary randomness" with a backward ("L") proposal,
    as in SMCP3 or RAVI.
    Returns (sampled_rgb, log_q_score).

    Specifically, this proposal works by proposing 3 RGB values:
    - One from a tight uniform around previous_rgb
    - One from a tight uniform around observed_rgb
    - One from a potentially broader uniform around the midpoint of the two.

    Then, one of these 3 RGB values is resampled and return.

    This process is a "forward" ("K") proposal, which has sampled (1) the chosen RGB value,
    (2) the index amoung [0, 1, 2] for which of the 3 proposals generated it, and (3)
    two additional RGB values.

    We then imagine having an "L" proposal which, given (1), proposes (2) and (3).
    To estimate the _marginal_ probability of K having proposed (1), marginalizing
    over the choice of (2) and (3), we can return the value log_K - log_L
    (where these terms are the log densities of the K and L proposals, respectively,
    evaluated at the values we sampled out of the K proposal).

    One remaining TODO: this proposal has no probability of generating a value that is far outside
    the range of the previous and observed values.  This means we technically do not have absolute continuity.
    In practice this means if the posterior ever assigns mass to RGB values outside this range, we can't
    propose traces that match that part of the posterior.
    """
    metadata = {}
    color_shift_scale = inference_hyperparams.effective_color_transition_scale
    d = 1 / (1 / color_shift_scale + 1 / color_scale)

    r_diff = jnp.abs(previous_rgb[0] - observed_rgb[0])
    g_diff = jnp.abs(previous_rgb[1] - observed_rgb[1])
    b_diff = jnp.abs(previous_rgb[2] - observed_rgb[2])
    diffs = jnp.array([r_diff, g_diff, b_diff])

    (k1, k2, k3) = split(key, 3)

    ## Proposal 1: near the previous value.
    min_rgbs1 = jnp.maximum(0.0, previous_rgb - diffs / 10 - 2 * d)
    max_rgbs1 = jnp.minimum(1.0, previous_rgb + diffs / 10 + 2 * d)
    proposed_rgb_1 = uniform.sample(k1, min_rgbs1, max_rgbs1)
    log_q_rgb_1 = uniform.logpdf(proposed_rgb_1, min_rgbs1, max_rgbs1)
    metadata["min_rgbs1"] = min_rgbs1
    metadata["max_rgbs1"] = max_rgbs1

    ## Proposal 2: near the observed value.
    min_rgbs2 = jnp.maximum(0.0, observed_rgb - diffs / 10 - 2 * d)
    max_rgbs2 = jnp.minimum(1.0, observed_rgb + diffs / 10 + 2 * d)
    proposed_rgb_2 = uniform.sample(k2, min_rgbs2, max_rgbs2)
    log_q_rgb_2 = uniform.logpdf(proposed_rgb_2, min_rgbs2, max_rgbs2)
    metadata["min_rgbs2"] = min_rgbs2
    metadata["max_rgbs2"] = max_rgbs2

    ## Proposal 3: somewhere in the middle
    mean_rgb = (previous_rgb + observed_rgb) / 2
    min_rgbs3 = jnp.maximum(0.0, mean_rgb - 8 / 10 * diffs - 2 * d)
    max_rgbs3 = jnp.minimum(1.0, mean_rgb + 8 / 10 * diffs + 2 * d)
    proposed_rgb_3 = uniform.sample(k3, min_rgbs3, max_rgbs3)
    log_q_rgb_3 = uniform.logpdf(proposed_rgb_3, min_rgbs3, max_rgbs3)
    metadata["min_rgbs3"] = min_rgbs3
    metadata["max_rgbs3"] = max_rgbs3

    ## Resample one of the values.

    proposed_rgbs = jnp.array([proposed_rgb_1, proposed_rgb_2, proposed_rgb_3])
    log_qs = jnp.array([log_q_rgb_1, log_q_rgb_2, log_q_rgb_3])
    metadata["log_qs"] = log_qs
    metadata["proposed_rgbs"] = proposed_rgbs

    scores = (
        jax.vmap(lambda rgb: score_attribute_assignment(rgb, visprob, dnrprob))(
            proposed_rgbs
        )
        - log_qs
    )
    normalized_scores = normalize_log_scores(scores)
    sampled_index = jax.random.categorical(key, normalized_scores)
    sampled_rgb = proposed_rgbs[sampled_index]
    log_K_score = log_qs.sum() + normalized_scores[sampled_index]
    metadata["normalized_scores"] = normalized_scores
    metadata["sampled_index"] = sampled_index
    metadata["sampled_rgb"] = sampled_rgb
    metadata["log_K_score"] = log_K_score

    ## "L proposal": given the sampled rgb, the L proposal proposes
    # an index for which of the 3 proposals may have produced this sample RGB,
    # and also proposes the other two RGB values.
    # Here, we need to compute the logpdf of this L proposal having produced
    # the values we sampled out of the K proposal.
    log_qs_for_this_rgb = jnp.array(
        [
            uniform.logpdf(sampled_rgb, min_rgbs1, max_rgbs1),
            uniform.logpdf(sampled_rgb, min_rgbs2, max_rgbs2),
            uniform.logpdf(sampled_rgb, min_rgbs3, max_rgbs3),
        ]
    )
    normalized_L_logprobs = normalize_log_scores(log_qs_for_this_rgb)

    # L score for proposing the index
    log_L_score_for_index = normalized_L_logprobs[sampled_index]

    # Also add in the L score for proposing the other two RGB values.
    # The L proposal over these values will just generate them from their prior.
    log_L_score_for_unused_values = jnp.sum(log_qs) - log_qs[sampled_index]

    # full L score
    log_L_score = log_L_score_for_index + log_L_score_for_unused_values
    metadata["log_L_score"] = log_L_score

    ## Compute the overall estimate of the marginal density of proposing `sampled_rgb`.
    overall_score = log_K_score - log_L_score
    metadata["overall_score"] = overall_score

    ## Return
    return sampled_rgb, overall_score, metadata


def propose_depth_scale(key, trace):
    """
    Propose a new global depth scale, conditioned upon the other values in `trace`.
    Returns (depth_scale, log_q) where `depth_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2 = split(key, 2)

    def score_depth_scale(k, depth_scale):
        newtr = update_field(k, trace, "depth_scale", depth_scale)
        return newtr.get_score()

    support = get_hypers(trace)["depth_scale_kernel"].support
    scores = jax.vmap(score_depth_scale, in_axes=(0, 0))(
        split(k1, len(support)), support
    )

    normalized_scores = normalize_log_scores(scores)
    index = jax.random.categorical(k2, normalized_scores)

    return support[index], normalized_scores[index]


def propose_color_scale(key, trace):
    """
    Propose a new global color scale, conditioned upon the other values in `trace`.
    Returns (color_scale, log_q) where `color_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2 = split(key, 2)

    def score_color_scale(k, color_scale):
        newtr = update_field(k, trace, "color_scale", color_scale)
        return newtr.get_score()

    support = get_hypers(trace)["color_scale_kernel"].support
    scores = jax.vmap(score_color_scale, in_axes=(0, 0))(
        split(k1, len(support)), support
    )

    normalized_scores = normalize_log_scores(scores)
    index = jax.random.categorical(k2, normalized_scores)

    return support[index], normalized_scores[index]


### Utils ###
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
