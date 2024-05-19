###
def model_weight_comp_manual(observed_rgbd, vertices_O, faces, vertex_colors, X_WO, X_WC, hyperparams=r.DEFAULT_HYPERPARAMS):
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)
        weights, attributes = r.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, hyperparams
        )
        tr, weight = likelihoods.mixture_rgbd_sensor_model.importance(
            jax.random.PRNGKey(0),
            genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd))),
            (weights, attributes, 3.0, 0.07, 0., 10.)
        )
        return weight
def pose_to_score_manual(pose):
    return model_weight_comp_manual(
            observed_rgbd, patch_vertices_P, patch_faces, patch_vertex_colors, pose, X_WC
        )
print(pose_to_score_manual(X_WP))
print(jax.grad(pose_to_score_manual)(X_WP))
print(jax.jit(jax.grad(pose_to_score_manual))(X_WP))






class LaplaceRGBDPixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Distribution over an RGBD value.  Applies a separate Laplace
    distribution to the RGB and depth values.
    Args:
    - rendered_rgbd (4,): the mean of the Laplace distributions (e.g. from a deterministic renderer)
    - color_scale (): the scale of the Laplace distribution (in RGB color space)
    - depth_scale (): the scale of the Laplace distribution (in depth space)
    Returns:
    - rgbd (4,)
    """
    def sample(self, key, rendered_rgbd, color_scale, depth_scale):
        rgb = laplace.sample(key, rendered_rgbd[:3], color_scale)
        depth = laplace.sample(key, rendered_rgbd[3], depth_scale)
        return jnp.concatenate([rgb, jnp.array([depth])])

    def logpdf(self, observed_rgbd, rendered_rgbd, color_scale, depth_scale):
        rgb_logpdf = laplace.logpdf(observed_rgbd[:3], rendered_rgbd[:3], color_scale)
        depth_logpdf = laplace.logpdf(observed_rgbd[3], rendered_rgbd[3], depth_scale)
        return rgb_logpdf + depth_logpdf

laplace_rgbd_pixel_model = LaplaceRGBDPixelModel()

class LaplaceRGB_UniformDepth_PixelModel(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    """
    Args:
    - rendered_rgbd (4,): the mean of the Laplace distributions (e.g. from a deterministic renderer)
    - color_scale (): the scale of the Laplace distribution (in RGB color space)
    Returns:
    - rgbd (4,)
    """
    def sample(self, key, rendered_rgbd, color_scale, depth_scale):
        rgb = genjax.sample(key, rendered_rgbd[:3], color_scale)
        depth = genjax.normal.sample(key, rendered_rgbd[3], depth_scale)
        return jnp.concatenate([rgb, jnp.array([depth])])

    def logpdf(self, observed_rgbd, rendered_rgbd, color_scale, depth_scale):
        rgb_logpdf = laplace.logpdf(observed_rgbd[:3], rendered_rgbd[:3], color_scale)
        depth_logpdf = genjax.normal.logpdf(observed_rgbd[3], rendered_rgbd[3], depth_scale)
        return rgb_logpdf + depth_logpdf

laplace_rgbd_pixel_model = LaplaceRGBDPixelModel()










l.multi_uniform_rgb_depth_laplace.sample(
    jax.random.PRNGKey(0),
    *a_
    # jnp.array([0.2, 0.8]),
    # jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]),
    # 0.1
)
a_ = [jtu.tree_map(lambda x: x[0, ...], a) if i < len(args)-1 else a for (i, a) in enumerate(fa)]

dist = l.ImageDistFromPixelDist(l.multi_uniform_rgb_depth_laplace, [True, True, False])
dist._vmap_in_axes()
args = (
    jnp.tile(jnp.array([0.2, 0.8]), (20, 30, 1)),
    jnp.tile(jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]), (20, 30, 1, 1)),
    depth_scale * jnp.ones(2)
)

# jax.vmap(
#     lambda key, *args: dist.pixel_dist.sample(key, *args),
#     in_axes=dist._vmap_in_axes()
# )(jax.random.split(jax.random.PRNGKey(0), 600), *dist._flattened_args(args))


# len(args)
# fa = 
# dist.sample(
#     jax.random.PRNGKey(0),
#     30, 20,
#     *args
# )
# [jtu.tree_map(lambda x: x.shape, x) for x in (jnp.arange(30 * 20), *fa) ]
