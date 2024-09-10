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
    k1, k2, k3, k4, k5 = split(key, 5)

    trace = update_field(k1, advanced_trace, "pose", pose)

    k2a, k2b = split(k2)
    depth_nonreturn_probs, log_q_dnrps, dnrp_metadata = propose_depth_nonreturn_probs(
        k2a, trace
    )
    trace = update_vmapped_field(
        k2b, trace, "depth_nonreturn_prob", depth_nonreturn_probs
    )

    k3a, k3b = split(k3)
    colors, visibility_probs, log_q_cvp = propose_colors_and_visibility_probs(
        k3a, trace, inference_hyperparams
    )
    trace = update_vmapped_fields(
        k3b, trace, ["colors", "visibility_prob"], [colors, visibility_probs]
    )
    log_q_cvp = 0.0

    k4a, k4b = split(k4)
    depth_scale, log_q_ds = propose_depth_scale(k4a, trace)
    trace = update_field(k4b, trace, "depth_scale", depth_scale)

    k5a, k5b = split(k5)
    color_scale, log_q_cs = propose_color_scale(k5a, trace)
    trace = update_field(k5b, trace, "color_scale", color_scale)

    log_q = log_q_dnrps + log_q_cvp + log_q_ds + log_q_cs
    return (
        trace,
        log_q,
        {"depth_nonreturn_proposal": dnrp_metadata, "dnrps": depth_nonreturn_probs},
    )


def propose_depth_nonreturn_probs(key, trace):
    """
    Propose a new depth nonreturn probability for every vertex, conditioned
    upon the other values in `trace`.
    Returns (depth_nonreturn_probs, log_q) where `depth_nonreturn_probs` is
    a vector of shape (n_vertices,) and `log_q` is (a fair estimate of)
    the log proposal density of this list of values.
    """
    observed_depths_per_points = PixelsPointsAssociation.from_hyperparams_and_pose(
        get_hypers(trace), get_new_state(trace)["pose"]
    ).get_point_depths(get_observed_rgbd(trace))

    depth_nonreturn_probs, per_vertex_log_qs, metadata = jax.vmap(
        propose_vertex_depth_nonreturn_prob, in_axes=(0, 0, 0, None, None, None)
    )(
        split(key, get_n_vertices(trace)),
        jnp.arange(get_n_vertices(trace)),
        observed_depths_per_points,
        get_prev_state(trace),
        get_new_state(trace),
        get_hypers(trace),
    )

    return depth_nonreturn_probs, per_vertex_log_qs.sum(), metadata


def propose_colors_and_visibility_probs(key, trace, inference_hyperparams):
    """
    Propose a new color and visibility probability for every vertex, conditioned
    upon the other values in `trace`.
    Returns (colors, visibility_probs, log_q) where `colors` has shape
    (n_vertices, 3), `visibility_probs` is a vector of shape (n_vertices,)
    and `log_q` is (a fair estimate of) the log proposal density of these
    values.
    """
    observed_rgbds_per_points = PixelsPointsAssociation.from_hyperparams_and_pose(
        get_hypers(trace), get_new_state(trace)["pose"]
    ).get_point_rgbds(get_observed_rgbd(trace))

    colors, visibility_probs, per_vertex_log_qs = jax.vmap(
        propose_vertex_color_and_visibility_prob,
        in_axes=(0, 0, 0, None, None, None, None),
    )(
        split(key, get_n_vertices(trace)),
        jnp.arange(get_n_vertices(trace)),
        observed_rgbds_per_points,
        get_prev_state(trace),
        get_new_state(trace),
        get_hypers(trace),
        inference_hyperparams,
    )

    return colors, visibility_probs, per_vertex_log_qs.sum()


def propose_vertex_depth_nonreturn_prob(
    key, vertex_index, observed_depth, previous_state, new_state, hyperparams
):
    """
    Propose a new depth nonreturn probability for the single vertex
    with index `vertex_index`.
    Returns (depth_nonreturn_prob, log_q) where `depth_nonreturn_prob` is
    the proposed value and `log_q` is (a fair estimate of) the log proposal density.
    """
    previous_dnrp = previous_state["depth_nonreturn_prob"][vertex_index]
    visibility_prob = new_state["visibility_prob"][vertex_index]
    latent_depth = new_state["pose"].apply(hyperparams["vertices"][vertex_index])[2]
    return _propose_vertex_depth_nonreturn_prob(
        key,
        observed_depth,
        latent_depth,
        visibility_prob,
        new_state["depth_scale"],
        previous_dnrp,
        hyperparams["depth_nonreturn_prob_kernel"],
        hyperparams["image_kernel"].get_depth_vertex_kernel(),
    )


def _propose_vertex_depth_nonreturn_prob(
    key,
    observed_depth,
    latent_depth,
    visibility_prob,
    depth_scale,
    previous_dnrp,
    dnrp_transition_kernel,
    obs_depth_kernel,
    return_metadata=True,
):
    def score_dnrp_value(dnrp_value):
        transition_score = dnrp_transition_kernel.logpdf(dnrp_value, previous_dnrp)
        likelihood_score = obs_depth_kernel.logpdf(
            observed_depth=observed_depth,
            latent_depth=latent_depth,
            depth_scale=depth_scale,
            visibility_prob=visibility_prob,
            depth_nonreturn_prob=dnrp_value,
        )
        return transition_score + likelihood_score

    support = dnrp_transition_kernel.support
    log_pscores = jax.vmap(score_dnrp_value)(support)
    log_normalized_scores = normalize_log_scores(log_pscores)
    index = jax.random.categorical(key, log_normalized_scores)
    # ^ since we are enumerating over every value in the domain, it is unnecessary
    # to add a 1/q score when resampling.  (Equivalently, we could include
    # q = 1/len(support), which does not change the resampling distribuiton at all.)

    if return_metadata:
        metadata = {
            "support": support,
            "log_normalized_scores": log_normalized_scores,
            "index": index,
            "observed_depth": observed_depth,
            "latent_depth": latent_depth,
            "prev_dnrp": previous_dnrp,
            "transition_score": jax.vmap(
                lambda dnrp_value: dnrp_transition_kernel.logpdf(
                    dnrp_value, previous_dnrp
                )
            )(support),
            "likelihood_score": jax.vmap(
                lambda dnrp_value: obs_depth_kernel.logpdf(
                    observed_depth,
                    latent_depth,
                    visibility_prob,
                    dnrp_value,
                    depth_scale,
                )
            )(support),
        }
    else:
        metadata = {}

    return support[index], log_normalized_scores[index], metadata


def propose_vertex_color_and_visibility_prob(
    key,
    vertex_index,
    observed_rgbd_for_this_vertex,
    previous_state,
    new_state,
    hyperparams,
    inference_hyperparams,
):
    """
    Propose a new color and visibility probability for the single vertex
    with index `vertex_index`.
    Returns (color, visibility_prob, log_q) where `color` and `visibility_prob`
    are the proposed values and `log_q` is (a fair estimate of) the log proposal density.
    """
    k1, k2 = split(key, 2)
    previous_rgb = previous_state["colors"][vertex_index]
    previous_visibility_prob = previous_state["visibility_prob"][vertex_index]
    latent_depth = new_state["pose"].apply(hyperparams["vertices"][vertex_index])[2]
    all_vis_probs = hyperparams["visibility_prob_kernel"].support

    def score_visprob_rgb(visprob, rgb):
        """
        Compute P(visprob, rgb, observed_rgbd_for_this_vertex | previous_visprob, previous_rgb).
        """
        rgb_transition_score = hyperparams["color_kernel"].logpdf(rgb, previous_rgb)
        visprob_transition_score = hyperparams["visibility_prob_kernel"].logpdf(
            visprob, previous_visibility_prob
        )
        likelihood_score = (
            hyperparams["image_kernel"]
            .get_rgbd_vertex_kernel()
            .logpdf(
                observed_rgbd=observed_rgbd_for_this_vertex,
                latent_rgbd=jnp.append(rgb, latent_depth),
                rgb_scale=new_state["color_scale"],
                depth_scale=new_state["depth_scale"],
                visibility_prob=visprob,
                depth_nonreturn_prob=new_state["depth_nonreturn_prob"][vertex_index],
            )
        )
        return rgb_transition_score + visprob_transition_score + likelihood_score

    # Propose a rgb value for each visprob.
    # `rgbs` has shape (len(all_vis_probs), 3).
    # `log_qs_rgb` has shape (len(all_vis_probs),).
    rgbs, log_qs_rgb = jax.vmap(
        lambda k, visprob: propose_vertex_color_given_visibility(
            k,
            visprob,
            observed_rgbd_for_this_vertex[:3],
            score_visprob_rgb,
            previous_rgb,
            new_state,
            hyperparams,
        )
    )(split(k1, len(all_vis_probs)), all_vis_probs)

    # shape: (len(all_vis_probs),)
    log_pscores = jax.vmap(score_visprob_rgb, in_axes=(0, 0))(all_vis_probs, rgbs)

    # We don't need to subtract a q score for the visibility probability, since
    # we are enumerating over every value in the domain.  (Equivalently,
    # we could subtract a log q score of log(1/len(support)) for each value.)
    log_weights = log_pscores - log_qs_rgb
    log_normalized_scores = normalize_log_scores(log_weights)
    index = jax.random.categorical(k2, log_normalized_scores)

    rgb = rgbs[index]
    visibility_prob = all_vis_probs[index]
    log_q_score = log_normalized_scores[index] + log_qs_rgb[index]

    return rgb, visibility_prob, log_q_score


def propose_vertex_color_given_visibility(
    key,
    visprob,
    observed_rgb,
    score_visprob_and_rgb,
    previous_rgb,
    new_state,
    hyperparams,
):
    """
    This samples an rgb value from a proposal which first proposes 3 different rgb values,
    then resamples one.  The log q score is estimated using
    simple logic for "filling in the auxiliary randomness" with a backward ("L") proposal,
    as in SMCP3 or RAVI.
    Returns (sampled_rgb, log_q_score).

    Specifically, this proposes 3 RGB values:
    - One from a tight uniform around previous_rgb
    - One from a tight uniform around observed_rgb
    - One from a potentially broader uniform around the midpoint of the two.

    One of these 3 RGB values is then resampled.

    This creates a "forward" ("K") proposal, which has sampled (1) the chosen RGB value,
    (2) the index amoung [0, 1, 2] for which of the 3 proposals generated it, and (3)
    two additional RGB values.

    We then imagine having an "L" proposal which, given (1), proposes (2) and (3).
    To estimate the probability of having proposed (1) alone, we return log_K - log_L.

    One remaining TODO: this proposal has no probability of generating a value that is far outside
    the range of the previous and observed values.  This means we technically do not have absolute continuity.
    In practice this means if the posterior ever assigns mass to RGB values outside this range, we can't
    propose traces that match that part of the posterior.
    """
    color_shift_scale = hyperparams["color_kernel"].scale
    color_scale = new_state["color_scale"]
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

    ## Proposal 2: near the observed value.
    min_rgbs2 = jnp.maximum(0.0, observed_rgb - diffs / 10 - 2 * d)
    max_rgbs2 = jnp.minimum(1.0, observed_rgb + diffs / 10 + 2 * d)
    proposed_rgb_2 = uniform.sample(k2, min_rgbs2, max_rgbs2)
    log_q_rgb_2 = uniform.logpdf(proposed_rgb_2, min_rgbs2, max_rgbs2)

    ## Proposal 3: somewhere in the middle
    mean_rgb = (previous_rgb + observed_rgb) / 2
    min_rgbs3 = jnp.maximum(0.0, mean_rgb - 8 / 10 * diffs - 2 * d)
    max_rgbs3 = jnp.minimum(1.0, mean_rgb + 8 / 10 * diffs + 2 * d)
    proposed_rgb_3 = uniform.sample(k3, min_rgbs3, max_rgbs3)
    log_q_rgb_3 = uniform.logpdf(proposed_rgb_3, min_rgbs3, max_rgbs3)

    ## Resample one of the values.

    proposed_rgbs = jnp.array([proposed_rgb_1, proposed_rgb_2, proposed_rgb_3])
    log_qs = jnp.array([log_q_rgb_1, log_q_rgb_2, log_q_rgb_3])

    scores = (
        jax.vmap(lambda rgb: score_visprob_and_rgb(visprob, rgb))(proposed_rgbs)
        - log_qs
    )
    normalized_scores = normalize_log_scores(scores)
    sampled_index = jax.random.categorical(key, normalized_scores)
    sampled_rgb = proposed_rgbs[sampled_index]
    log_K_score = log_qs.sum() + normalized_scores[sampled_index]

    ## "L proposal": given the sampled rgb, estimate the probability that
    # it came from the one of the 3 proposals that actually was used.
    log_qs_for_this_rgb = jnp.array(
        [
            uniform.logpdf(sampled_rgb, min_rgbs1, max_rgbs1),
            uniform.logpdf(sampled_rgb, min_rgbs2, max_rgbs2),
            uniform.logpdf(sampled_rgb, min_rgbs3, max_rgbs3),
        ]
    )
    normalized_L_logprobs = normalize_log_scores(log_qs_for_this_rgb)

    # L score for proposing the index
    log_L_score = normalized_L_logprobs[sampled_index]

    # Also add in the L score for proposing the other two RGB values.
    # The L proposal over these values will just generate them from their prior.
    log_L_score += jnp.sum(log_qs) - log_qs[sampled_index]

    ## Compute the overall score.
    overall_score = log_K_score - log_L_score

    ## Return
    return sampled_rgb, overall_score


def propose_depth_scale(key, trace):
    """
    Propose a new global depth scale, conditioned upon the other values in `trace`.
    Returns (depth_scale, log_q) where `depth_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    # Placeholder
    return get_prev_state(trace)["depth_scale"], 0.0


def propose_color_scale(key, trace):
    """
    Propose a new global color scale, conditioned upon the other values in `trace`.
    Returns (color_scale, log_q) where `color_scale` is the proposed value and
    `log_q` is (a fair estimate of) the log proposal density.
    """
    # Placeholder
    return get_prev_state(trace)["color_scale"], 0.0


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
