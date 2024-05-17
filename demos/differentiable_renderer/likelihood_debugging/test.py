### Preliminaries ###

import jax.numpy as jnp
import jax
import os
import trimesh
import b3d
from b3d import Pose
import rerun as rr
import genjax
from tqdm import tqdm
import demos.differentiable_renderer.likelihood_debugging.demo_utils as du
import demos.differentiable_renderer.likelihood_debugging.model as m
import demos.differentiable_renderer.likelihood_debugging.likelihoods as l
import b3d.differentiable_renderer as r



### Preliminaries ###
(
    renderer,
    (observed_rgbds, gt_rots),
    ((patch_vertices_P, patch_faces, patch_vertex_colors), X_WP),
    X_WC
) = du.get_renderer_boxdata_and_patch()

hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)

depth_scale = 0.1
mindepth = -1.0
maxdepth = 2.0
# likelihood = l.ArgMap(
#     l.ImageDistFromPixelDist(l.multi_uniform_rgb_depth_laplace, [True, True, False]),
#     lambda weights, rgbds: (
#         renderer.height, renderer.width,
#         l.normalize(weights[..., 1:]),
#         rgbds,
#         depth_scale
#     )
# )
likelihood = l.ArgMap(
    l.ImageDistFromPixelDist(
        l.mixture_of_uniform_and_multi_uniformrgb_laplacedepth,
        [True, True, False, False, False]
    ),
    lambda weights, rgbds: (
        renderer.height, renderer.width,
        weights,
        rgbds,
        depth_scale, mindepth, maxdepth
    )
)
likelihood.sample(
    jax.random.PRNGKey(0),
    jnp.tile(jnp.array([0.2, 0.3, 0.5]), (renderer.height, renderer.width, 1)),
    jnp.tile(jnp.array([[0.5, 0.5, 0.5, 1.5], [0.2, 0.2, 0.2, 1.0]]), (renderer.height, renderer.width, 1, 1))
)

model = m.single_object_model_factory(
    renderer,
    likelihood,
    hyperparams,
    lambda weights, attributes, likelihood_args: (
        r.dist_params_to_average(weights, attributes, jnp.zeros(4))
    )
)

key = jax.random.PRNGKey(2)
trace = model.simulate(
    key, (patch_vertices_P, patch_faces, patch_vertex_colors, ())
)
trace.get_score()

observed_rgbd, weights, attributes = trace.get_retval()

trace, weight = model.importance(
    key,
    genjax.choice_map({
        "pose": X_WP,
        "camera_pose": X_WC,
        "observed_rgbd": observed_rgbds[0]
    }),
    (patch_vertices_P, patch_faces, patch_vertex_colors, ())
)

### Generate image samples ###
rr.init("test")
rr.connect("127.0.0.1:8812")
rr.log("/trace/rgb/observed", rr.Image(trace.get_retval()[0][:, :, :3]))

### Grid over positions and show samples ###