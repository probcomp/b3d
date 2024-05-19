import jax
import jax.numpy as jnp
import genjax
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as rendering
import b3d
from b3d import Pose
from b3d.model import uniform_pose
import rerun as rr
import demos.differentiable_renderer.tracking.utils as utils

def normalize(v):
    return v / jnp.sum(v)

def model_singleobject_gl_factory(renderer,
                                  lab_noise_scale=3.0,
                                  depth_noise_scale=0.07,
                                  outlier_prob = 0.05,
                                  hyperparams = rendering.DEFAULT_HYPERPARAMS,
                                  mindepth=0.,
                                  maxdepth=20.
                                ):
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors):
        X_WO = uniform_pose(jnp.ones(3)*-10.0, jnp.ones(3)*10.0) @ "pose"
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
        
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, hyperparams
        )
        # Increase the probability that pixel values are drawn from uniform by the outlier_prob.
        # (It will already be high if the pixel only hit the background.)
        weights = weights.at[:, :, 0].set(weights[:, :, 0] + outlier_prob)
        weights = jax.vmap(normalize, in_axes=(0,))(weights.reshape(-1, weights.shape[-1])).reshape(weights.shape)

        observed_rgbd = likelihoods.mixture_rgbd_sensor_model(
            weights, attributes,
            lab_noise_scale, depth_noise_scale, mindepth, maxdepth
        ) @ "observed_rgbd"
        average_rgbd = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
        return (observed_rgbd, average_rgbd)
    return model

def rr_viz_trace(trace, renderer):
    (observed_rgbd, average_rgbd) = trace.get_retval()
    # rr.log("/trace/vertices", rr.Points3D(positions=vertices, colors=jnp.ones_like(vertices), radii=0.001))
    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/rgb/rendered_average", rr.Image(average_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    rr.log("/trace/depth/rendered_average", rr.DepthImage(average_rgbd[:, :, 3]))
    
    (vertices, faces, vertex_colors) = trace.get_args()
    pose = trace["pose"]
    cam_pose = trace["camera_pose"]
    rr.log("/trace/mesh", rr.Mesh3D(
        vertex_positions=pose.apply(vertices),
        indices=faces,
        vertex_colors=vertex_colors
    ))
    rr.log("/trace/camera",
        rr.Pinhole(
            focal_length=renderer.fx,
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
            )
        )
    xyzs_C = utils.unproject_depth(observed_rgbd[:, :, 3], renderer)
    xyzs_W = cam_pose.apply(xyzs_C)
    rr.log("/3D/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )

    rr.log("/trace/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()))

### Multi object model ###
def model_multiobject_gl_factory(renderer,
                                  lab_noise_scale=3.0,
                                  depth_noise_scale=0.07,
                                  outlier_prob = 0.05,
                                  hyperparams = rendering.DEFAULT_HYPERPARAMS,
                                  mindepth=0.,
                                  maxdepth=20.
                                ):
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors):
        #    (N, V, 3)    (N, F, 3)   (N, V, 3)
        # where N = num objects
        N = vertices_O.shape[0]

        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"

        Xs_WO = genjax.map_combinator(in_axes=(0, 0))(uniform_pose)(
            jnp.ones((N, 3))*-10.0, jnp.ones((N, 3))*10.0
        ) @ "poses"
        
        vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W.reshape(-1, 3))

        v = vertices_C.reshape(-1, 3)
        vc = vertex_colors.reshape(-1, 3)

        # shift up each object's face indices by the number of vertices in the previous objects
        f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
        f = f.reshape(-1, 3)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, v, f, vc, hyperparams
        )
        # Increase the probability that pixel values are drawn from uniform by the outlier_prob.
        # (It will already be high if the pixel only hit the background.)
        weights = weights.at[:, :, 0].set(weights[:, :, 0] + outlier_prob)
        weights = jax.vmap(normalize, in_axes=(0,))(weights.reshape(-1, weights.shape[-1])).reshape(weights.shape)

        observed_rgbd = likelihoods.mixture_rgbd_sensor_model(
            weights, attributes,
            lab_noise_scale, depth_noise_scale, mindepth, maxdepth
        ) @ "observed_rgbd"
        average_rgbd = rendering.dist_params_to_average(weights, attributes, jnp.zeros(4))
        return (observed_rgbd, average_rgbd)
    return model

def rr_viz_multiobject_trace(trace, renderer):
    (observed_rgbd, average_rgbd) = trace.get_retval()

    rr.log("/trace/rgb/observed", rr.Image(observed_rgbd[:, :, :3]))
    rr.log("/trace/rgb/rendered_average", rr.Image(average_rgbd[:, :, :3]))
    rr.log("/trace/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]))
    rr.log("/trace/depth/rendered_average", rr.DepthImage(average_rgbd[:, :, 3]))
    
    (vertices_O, faces, vertex_colors) = trace.get_args()
    Xs_WO = trace.strip()["poses"].inner.value # TODO: do this better
    vertices_W = jax.vmap(lambda X_WO, v_O: X_WO.apply(v_O), in_axes=(0, 0))(Xs_WO, vertices_O)
    N = vertices_O.shape[0]
    f = jax.vmap(lambda i, f: f + i*vertices_O.shape[1], in_axes=(0, 0))(jnp.arange(N), faces)
    f = f.reshape(-1, 3)

    cam_pose = trace["camera_pose"]
    rr.log("/trace/mesh", rr.Mesh3D(
        vertex_positions=vertices_W.reshape(-1, 3),
        indices=f,
        vertex_colors=vertex_colors.reshape(-1, 3)
    ))
    rr.log("/trace/camera",
        rr.Pinhole(
            focal_length=renderer.fx,
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
            )
        )
    rr.log("/trace/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()))
    xyzs_C = utils.unproject_depth(observed_rgbd[:, :, 3], renderer)
    xyzs_W = cam_pose.apply(xyzs_C)
    rr.log("/3D/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0]))
    )

# patch_vertices_W = X_WP.inv().apply(patch_vertices_P)
# vertices = patch_vertices_W
# faces = patch_faces
# vertex_colors = patch_vertex_colors
# alpha = 3.0
# vetex_colors_with_alpha = jnp.concatenate([vertex_colors, alpha*jnp.ones((vertex_colors.shape[0], 1))], axis=1)


# rr.log("mymesh", rr.Mesh3D(
#     vertex_positions=vertices,
#     indices=faces,
#     vertex_colors=vetex_colors_with_alpha
# ))
