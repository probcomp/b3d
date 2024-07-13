from functools import partial
from jax.scipy.special import logsumexp


def make_camera_scorer(observation_model_logpdf, particle_proposal):
    """
    Args:

        observation_model_logpdf: Function that takes (y, x, cam, intr) and 
            returns logP(y| cam, x)

        particle_proposal: Sampler that takes (key, ys[:T,i], cams[:T], intr) and 
            returns a tuple (x[i], logQ(x[i])), containing a scored particle 
            proposal.

    We assume both the observation model and proposal are independent with respect to each particle.
    We assume p(c) and p(x) are constant.
    """
    obs_model_mapped_over_i  = jax.vmap(observation_model_logpdf, (0, 0, None, None, None))
    obs_model_mapped_over_ti = jax.vmap(obs_model_mapped_over_i, (0, None, 0, None, None))

    proposal_mapped_over_i   = jax.vmap(particle_proposal, (0, 1, None, None))
    proposal_mapped_over_si  = jax.vmap(proposal_mapped_over_i, (0,None,None,None))

    def camera_score(key, ys, cams, intr, sig, S):
        
        T = ys.shape[0]
        N = ys.shape[1]

        # Branch and get random keys 
        # for the particle proposals
        key = jax.random.split(key)[1]
        keys = jax.random.split(key, (S,N))

        # Shapes should be (S,N,3) and (S,N)
        xs, log_qxs = proposal_mapped_over_si(keys, ys, cams, intr)

        # Shape should be (S, T, N)
        log_pys = jax.vmap(obs_model_mapped_over_ti, (None,0,None,None, None))(ys, xs, cams, intr, sig)
        
        return logsumexp(log_pys.sum(1), axis=0).sum() - N*jnp.log(S)


    @partial(jax.jit, static_argnames=("S",))
    def particle_inference(key, ys, cams, intr, sig, S):

        T = ys.shape[0]
        N = ys.shape[1]

        # Branch and get random keys 
        # for the particle proposals
        key = jax.random.split(key)[1]
        keys = jax.random.split(key, (S,N))

        # Shapes should be (S,N,3) and (S,N)
        xs, log_qxs = proposal_mapped_over_si(keys, ys, cams, intr)

        # Shape should be (S, T, N)
        log_pys = jax.vmap(obs_model_mapped_over_ti, (None,0,None,None,None))(ys, xs, cams, intr, sig)

        scores = log_pys.sum(1)
        ii = jnp.argmax(scores, axis=0)
        xs_winner = xs[ii,jnp.arange(N)]

        return xs_winner 


    return camera_score, particle_inference



from jax.scipy.stats.norm import logpdf as normal_logpdf


def particle_proposal_0(key, ys_Tx2, cams_T, intr):
    t = 0
    y = ys_Tx2[t]
    cam = cams_T[t]

    z = jax.random.uniform(key, minval=intr.near, maxval = intr.far)
    x = cam(camera_from_screen_and_depth(y, z, intr))

    return x, 0.0

def particle_proposal_1(key, ys_Tx2, cams_T, intr):
    t = 1
    y = ys_Tx2[t]
    cam = cams_T[t]

    z = jax.random.uniform(key, minval=intr.near, maxval = intr.far)
    x = cam(camera_from_screen_and_depth(y, z, intr))

    return x, 0.0

def observation_model_logpdf(y, x, cam, intr, sig=10.):
    y_ = screen_from_world(x, cam, intr, culling=True) 
    logp = normal_logpdf(y - y_, 0., sig).sum()
    # logp = - jnp.linalg.norm(y - y_)
    return jnp.clip(logp, -1e6, jnp.inf)
                    
    


camera_score, particle_inference = make_camera_scorer(observation_model_logpdf, particle_proposal_0)

intr = Intrinsics(*_intr[:-2], 1e-1,10.0)
ys = jnp.stack([uvs0, uvs1], axis=0)

vmap_camera_score = partial(jax.jit, static_argnames=("S",))(jax.vmap(
    lambda key, ys, cam, intr, sig, S: 
        camera_score(key, ys, Pose.stack_poses([Pose.id(), cam]), intr, sig, S), (0, None, 0, None, None, None)
))
