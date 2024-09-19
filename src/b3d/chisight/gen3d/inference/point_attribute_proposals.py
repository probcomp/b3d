import jax
import jax.numpy as jnp
import jax.random
from jax.random import split

from b3d.modeling_utils import renormalized_color_laplace

from ..image_kernel import (
    PixelsPointsAssociation,
    calculate_latent_and_observed_correspondences,
)
from ..model import (
    get_hypers,
    get_n_vertices,
    get_new_state,
    get_observed_rgbd,
    get_prev_state,
)
from .utils import all_pairs, normalize_log_scores


def propose_all_pointlevel_attributes(key, trace, inference_hyperparams):
    """
    Propose a new color, visibility probability, and depth non-return probability
    for every vertex, conditioned upon the other values in `trace`.

    Returns (colors, visibility_probs, depth_nonreturn_probs, log_q, metadata),
    where colors has shape (n_vertices, 3), visibility_probs and depth_nonreturn_probs
    have shape (n_vertices,), log_q (a float) is (an estimate of)
    the overall log proposal density, and metadata is a dict.
    """
    if inference_hyperparams.in_inference_only_assoc_one_point_per_pixel:
        observed_rgbds_for_registered_points, _, _, _, vertex_indices = (
            calculate_latent_and_observed_correspondences(
                get_observed_rgbd(trace), get_new_state(trace), get_hypers(trace)
            )
        )
        observed_rgbds_per_point = -jnp.ones((get_n_vertices(trace), 4))
        observed_rgbds_per_point = observed_rgbds_per_point.at[vertex_indices].set(
            observed_rgbds_for_registered_points, mode="drop"
        )
    else:
        observed_rgbds_per_point = PixelsPointsAssociation.from_hyperparams_and_pose(
            get_hypers(trace), get_new_state(trace)["pose"]
        ).get_point_rgbds(get_observed_rgbd(trace))

    sample, metadata = jax.vmap(
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

    return (
        sample["colors"],
        sample["visibility_prob"],
        sample["depth_nonreturn_prob"],
        metadata["log_q_score"].sum(),
        metadata,
    )


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
        latent_rgbd_for_point=jnp.array(
            [
                0.0,
                0.0,
                0.0,
                new_state["pose"].apply(hyperparams["vertices"][vertex_index])[2],
            ]
        ),
        previous_color=prev_state["colors"][vertex_index],
        previous_visibility_prob=prev_state["visibility_prob"][vertex_index],
        previous_dnrp=prev_state["depth_nonreturn_prob"][vertex_index],
        color_scale=new_state["color_scale"],
        depth_scale=new_state["depth_scale"],
        hyperparams=hyperparams,
        inference_hyperparams=inference_hyperparams,
    )


def _propose_a_points_attributes(
    key,
    observed_rgbd_for_point,
    latent_rgbd_for_point,
    previous_color,
    previous_visibility_prob,
    previous_dnrp,
    color_scale,
    depth_scale,
    hyperparams,
    inference_hyperparams,
):
    k1, k2 = split(key, 2)
    dnrp_transition_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    visibility_transition_kernel = hyperparams["visibility_prob_kernel"]
    color_kernel = hyperparams["color_kernel"]
    obs_rgbd_kernel = hyperparams["image_kernel"].get_rgbd_vertex_kernel()
    latent_depth = latent_rgbd_for_point[3]
    intrinsics = hyperparams["intrinsics"]

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
            intrinsics=intrinsics,
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
        {
            "colors": rgb,
            "visibility_prob": visibility_prob,
            "depth_nonreturn_prob": dnr_prob,
        },
        {
            "log_q_score": log_q_score,
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
    inference_hyperparams,
    color_kernel,
):
    k1, _ = split(key)
    value_if_observed_is_valid, log_q_if_valid, metadata_if_valid = (
        propose_vertex_color_given_other_attributes_for_valid_observed_rgb(
            k1,
            visprob,
            dnrprob,
            observed_rgb,
            score_attribute_assignment,
            previous_rgb,
            inference_hyperparams,
        )
    )
    value_if_observed_is_invalid = previous_rgb  # color_kernel.sample(k2, previous_rgb)
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
    inference_hyperparams,
):
    """
    This samples an rgb value from a proposal which first proposes 3 different rgb values,
    then resamples one.  The log q score is estimated using
    simple logic for "filling in the auxiliary randomness" with a backward ("L") proposal,
    as in SMCP3 or RAVI.
    Returns (sampled_rgb, log_q_score).

    Specifically, this proposal works by proposing 2 RGB values:
    - One from a laplace around previous_rgb
    - One from a laplace around observed_rgb

    Then, one of these 2 RGB values is resampled and return.

    This process is a "forward" ("K") proposal, which has sampled (1) the chosen RGB value,
    (2) the index amoung [0, 1] for which of the 2 proposals generated it, and (3)
    one additional RGB values.

    We then imagine having an "L" proposal which, given (1), proposes (2) and (3).
    To estimate the _marginal_ probability of K having proposed (1), marginalizing
    over the choice of (2) and (3), we can return the value log_K - log_L
    (where these terms are the log densities of the K and L proposals, respectively,
    evaluated at the values we sampled out of the K proposal).
    """
    metadata = {}

    k1, k2 = split(key, 2)

    ## Proposal 1: near the previous value.
    scale1 = inference_hyperparams.prev_color_proposal_laplace_scale
    proposed_rgb_1 = renormalized_color_laplace.sample(k1, previous_rgb, scale1)
    proposed_rgb_1 = jnp.where(
        inference_hyperparams.do_stochastic_color_proposals,
        proposed_rgb_1,
        previous_rgb,
    )
    log_q_rgb_1 = renormalized_color_laplace.logpdf(
        proposed_rgb_1, previous_rgb, scale1
    )

    ## Proposal 1: near the observed value.
    scale2 = inference_hyperparams.obs_color_proposal_laplace_scale
    proposed_rgb_2 = renormalized_color_laplace.sample(k2, observed_rgb, scale2)
    proposed_rgb_2 = jnp.where(
        inference_hyperparams.do_stochastic_color_proposals,
        proposed_rgb_2,
        observed_rgb,
    )
    log_q_rgb_2 = renormalized_color_laplace.logpdf(
        proposed_rgb_2, observed_rgb, scale2
    )

    ## Resample one of the values.

    proposed_rgbs = jnp.array([proposed_rgb_1, proposed_rgb_2])
    log_qs = jnp.array([log_q_rgb_1, log_q_rgb_2])
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
    metadata["normalized_scores"] = normalized_scores
    metadata["sampled_index"] = sampled_index
    metadata["sampled_rgb"] = sampled_rgb

    log_K_score = log_qs.sum() + normalized_scores[sampled_index]
    metadata["log_K_score"] = log_K_score

    ## "L proposal": given the sampled rgb, the L proposal proposes
    # an index for which of the 3 proposals may have produced this sample RGB,
    # and also proposes the other two RGB values.
    # Here, we need to compute the logpdf of this L proposal having produced
    # the values we sampled out of the K proposal.
    log_qs_for_this_rgb = jnp.array(
        [
            renormalized_color_laplace.logpdf(sampled_rgb, previous_rgb, scale1),
            renormalized_color_laplace.logpdf(sampled_rgb, observed_rgb, scale2),
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
    # overall_score = normalized_scores[sampled_index]  # + log_qs["sampled_index"]
    metadata["overall_score"] = overall_score

    ## Return
    return sampled_rgb, overall_score, metadata
