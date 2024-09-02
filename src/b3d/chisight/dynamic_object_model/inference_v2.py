### TODO: pass PRNGKey through all these functions.


def inference_step(old_trace, observation):
    advanced_trace = advance_time(old_trace, observation)

    # Propose N new traces, and get an importance weight
    # for each.
    #
    # One way to think about this: N times, propose a pose,
    # and then propose set all the other parameters to a good
    # value given that pose.
    #
    # Note that if we later want to stratify (=~ grid over)
    # the set of poses we consider, rather than proposing each
    # one independently at random, this can
    # be done soundly and with very little change to the code.
    trs, weights = jax.vmap(lambda _: update_trace(advanced_trace))(jnp.arange(N))

    # Resample one of the traces proportionally to its weight.
    idx = genjax.categorical(weights)

    # If we want to do SMC, the overall SMC incremental weight
    # for this step is
    overall_incremental_weight = logmeanexp(weights)
    # ^ One way of understanding this is that it is just
    # an estimate of P(observation_T | inferred_state_{T-1}).
    # It may be useful to have, since it is a well-behaved score
    # which we can check with an inference controller,
    # to decide if more compute is needed to find a good trace.

    return trs[idx], overall_incremental_weight


def update_trace(advanced_trace):
    """
    This returns (tr, weight).

    tr contains a full new state, s_T, as well
    as the observation o_T, and the previous latents (in
    the args), s_{T-1}.

    weight is an estimate of
        p(s_T, o_T | s_{T-1}) / q(s_T ; o_T, s_{T-1})
    where `q` is the distribution that this `update_trace`
    function samples from when generating s_T.

    (Fun fact: this p/q term is itself an estimate of p(o_T | s_{T-1}).)
    """
    proposed_trace, log_q = propose_update(advanced_trace)
    log_p = proposed_trace.get_score()
    return proposed_trace, log_p - log_q


def propose_update(advanced_trace):
    total_logq = 0

    # Propose a pose `pose` and get the probability
    # of having proposed this pose.
    pose, log_q_pose = sample_pose(advanced_trace["pose"])
    tr_with_pose = genjax.update(advanced_trace, pose)
    total_logq += log_q_pose

    # Propose a new depth outlier probability.
    # Return the trace, updated to have this outlier value,
    # and the probability of the proposal having chosen
    # this value.
    tr_with_depth_outlier, log_q_depth_outlier = propose_depth_outlier_probability(
        tr_with_pose
    )
    total_logq += log_q_depth_outlier

    # Propose a new color outlier probability, AND new
    # point colors.
    # Return the trace, updated to have these new values,
    # and an estimate of the probability of the proposal
    # having chosen these values.
    tr_with_colors, log_q_colors = propose_colors_and_color_outlier_probability(
        tr_with_depth_outlier
    )
    total_logq += log_q_colors

    # Now propose an updated depth variance value,
    # and after that, propose an updated color variance value.
    tr_with_depth_var, log_q_depthvar = propose_depth_var(tr_with_colors)
    tr_with_color_var, log_q_colorvar = propose_color_var(tr_with_depth_var)
    total_logq += log_q_depthvar
    total_logq += log_q_colorvar

    return tr_with_color_var, total_logq


def sample_pose(previous_pose):
    # Sample from gaussian VMF.
    # Return sampled value, and its logpdf.
    pass


def propose_depth_var(tr):
    # Enumerative update.
    # For each possible depth variance value,
    # genjax.update the trace.
    # Resample one of the options according to the weights.
    # Return the chosen trace.
    # Return log q = log (weight of chosen trace / sum of update weights)
    pass


def propose_color_var(tr):
    # Enumerative update.
    # For each possible depth variance value,
    # genjax.update the trace.
    # Resample one of the options according to the weights.
    # Return the chosen trace.
    # Return log q = log (weight of chosen trace / sum of update weights)
    pass


def propose_depth_outlier_probability(tr):
    # Propose each point's outlier probability independently in parallel.
    # The total proposal probability is the product of the individual proposal
    # probabilities (hence, a sum in log space).
    chosen_values, log_q_scores = jax.vmap(
        lambda point_idx: propose_depth_outlier_probability_for_point(tr, point_idx)
    )(jnp.arange(N_pts))
    new_tr = genjax.update(tr, chosen_values)
    return new_tr, log_q_scores.sum()


def propose_colors_and_color_outlier_probability(tr):
    # Propose each point's outlier probability, and new color, independently
    # in parallel.
    chosen_outlier_values, chosen_color_values, log_q_scores = jax.vmap(
        lambda point_idx: propose_color_and_color_outlier_probability_for_point(
            tr, point_idx
        )
    )(jnp.arange(N_pts))
    new_tr = genjax.update(tr, chosen_outlier_values, chosen_color_values)
    return new_tr, log_q_scores.sum()


def propose_depth_outlier_probability_for_point(tr, point_idx):
    # "dop" = "depth outlier prob"

    # Get a function which accepts a depth outlier prob for
    # point `point_idx` as input, and outputs the log probability
    # of the observed image given the latent state,
    # assuming the latent state is updated with this DOP.
    dop_to_log_likelihood = get_map_from_dop_to_likelihood_value(tr, point_idx)

    # Also assume we have access to a function `dop_prior`
    # which maps from a depth outlier probability to its prior probability,
    # given the previous depth outlier prob for this point.
    dop_to_prior_probability = get_map_from_dop_to_transition_probability(tr, point_idx)

    # Now -- grid over all possibilities.
    loglikelihoods = jax.vmap(dop_to_log_likelihood)(all_possible_dop_values)
    log_prior_scores = jax.vmap(dop_to_prior_probability)(all_possible_dop_values)
    total_scores = loglikelihoods + log_prior_scores

    # Resample one option based on its joint probability.
    normalized_scores = total_scores - logsumexp(total_scores)
    idx = genjax.categorical(normalized_scores)
    log_q = normalized_scores[idx]

    return all_possible_dop_values[idx], log_q


def propose_color_and_color_outlier_probability_for_point(tr, point_idx):
    # "cop" = "color outlier prob"

    # Assume we have access to three functions:
    # 1. Maps a `cop` -> log transition probability of having `cop` at this step
    # 2. Maps a `color` -> log transition probability of having `color` at this step
    # 3. Maps `(cop, color)` -> log likelihood of observed image, given `cop` and `color`
    cop_to_prior = get_cop_to_prior(tr, point_idx)
    color_to_prior = get_color_to_prior(tr, point_idx)
    cop_color_to_likelihood = get_cop_color_to_likelihood(tr, point_idx)

    # For every possible color outlier probability, consider updating to that
    # color outlier probability.  Propose a point color, given that color outlier prob.
    colorvals, logq_of_colorvals = jax.vmap(
        lambda cop: propose_color_for_point_given_color_outlier_probability(
            tr, point_idx, cop
        )
    )(all_possible_cop_values)

    ## Now, compute the scores for each update, and resample one.
    prior_cop_probs = jax.vmap(cop_to_prior)(all_possible_cop_values)
    prior_color_probs = jax.vmap(color_to_prior)(colorvals)
    likelihoods = jax.vmap(cop_color_to_likelihood, in_axes=(0, 0))(
        all_possible_cop_values, colorvals
    )
    p_scores = prior_cop_probs + prior_color_probs + likelihoods

    resampling_weights = p_scores - logq_of_colorvals
    normalized_log_weights = resampling_weights - logsumexp(resampling_weights)

    idx = genjax.categorical(normalized_log_weights)
    cop_to_return, color_to_return = all_possible_cop_values[idx], colorvals[idx]

    # To propose this, we needed to (1) propose this COP and (2)
    # propose this specific color val for that COP.
    overall_logq = normalized_log_weights[idx] + logq_of_colorvals[idx]

    return (cop_to_return, color_to_return, overall_logq)


def propose_color_for_point_given_color_outlier_probability(
    tr, point_idx, color_outlier_prob
):
    """
    Returns a pair (new_color, log_q), where
    `log_q` is an estimate of the probability of having proposed the new color.
    """
    # TODO: design this function with Nishad.
    # One idea: propose one value around the old color and propose
    # one value around the observed color, and resample one of these two
    # options.
    pass


###
def propose_depth_outlier_probability_for_point(tr, point_idx):
    # "dop" = "depth outlier prob"

    # Get a function which accepts a depth outlier prob for
    # point `point_idx` as input, and outputs the log probability
    # of the observed image given the latent state,
    # assuming the latent state is updated with this DOP.
    dop_to_log_likelihood = get_map_from_dop_to_likelihood_value(tr, point_idx)

    # Also assume we have access to a function `dop_prior`
    # which maps from a depth outlier probability to its prior probability,
    # given the previous depth outlier prob for this point.
    dop_to_prior_probability = get_map_from_dop_to_transition_probability(tr, point_idx)

    # Now -- grid over all possibilities.
    loglikelihoods = jax.vmap(dop_to_log_likelihood)(all_possible_dop_values)
    log_prior_scores = jax.vmap(dop_to_prior_probability)(all_possible_dop_values)
    total_scores = loglikelihoods + log_prior_scores

    # Resample one option based on its joint probability.
    normalized_scores = total_scores - logsumexp(total_scores)
    idx = genjax.categorical(normalized_scores)
    log_q = normalized_scores[idx]

    return all_possible_dop_values[idx], log_q


###
