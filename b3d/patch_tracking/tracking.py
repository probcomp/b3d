import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
import genjax
import b3d.differentiable_renderer as r
import optax

def all_pairs_2(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(X, Y), axis=-1),
        0, 1
    ).reshape(-1, 2)

def get_patches(centers, rgbs, xyzs_W, X_WC, fx):
    xyzs_C = X_WC.inv().apply(xyzs_W)
    def get_patch(center):
        center_x, center_y = center[0], center[1]
        del_pix = 3
        patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix-1,2*del_pix-1,3)).reshape(-1,3)
        patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix-1,2*del_pix-1,3)).reshape(-1,3)
        patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
            patch_points_C, patch_rgbs, patch_points_C[:,2] / fx * 2.0
    )
        X_CP = Pose.from_translation(patch_vertices_C.mean(0))
        X_WP = X_WC @ X_CP
        patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
        return (patch_vertices_P, patch_faces, patch_vertex_colors, X_WP)

    return jax.vmap(get_patch, in_axes=(0,))(centers)

def get_patches_with_default_centers(rgbs, xyzs_W, X_WC, fx):
    width_gradations = jnp.arange(44, 84, 6)
    height_gradations = jnp.arange(38, 96, 6)
    centers = all_pairs_2(height_gradations, width_gradations)
    return get_patches(centers, rgbs, xyzs_W, X_WC, fx)

def get_patch_tracker(model, patch_vertices_P, patch_faces, patch_vertex_colors, X_WC=Pose.identity()):
    """
    Returns:
    - get_initial_tracker_state:
        A function from the initial patch poses, Xs_WP, to an initial state object `tracker_state` for the patch tracker.
    - update_tracker_state:
        A function from `(tracker_state, new_observed_rgbd)` to a tuple `(new_patch_poses, updated_tracker_state)`,
        where `new_patch_poses` is a pair (positions, quaternions) for the updated patch poses,
        and `new_tracker_state` is the updated tracker state.
        At time 0, `new_observed_rgbd` should be the observed_rgbd for the frame at t=0.
    """
    @jax.jit
    def importance_from_pos_quat(key, positions, quaternions, observed_rgbd):
        poses = jax.vmap(lambda pos, quat: Pose.from_vec(jnp.concatenate([pos, quat])), in_axes=(0, 0))(positions, quaternions)
        trace, weight = model.importance(
            key,
            genjax.choice_map({
                "poses": genjax.vector_choice_map(genjax.choice(poses)),
                "camera_pose": X_WC,
                "observed_rgbd": observed_rgbd
            }),
            (patch_vertices_P, patch_faces, patch_vertex_colors, ())
        )
        return trace, weight
    
    def weight_from_pos_quat(pos, quat, observed_rgbd):
        return importance_from_pos_quat(pos, quat, observed_rgbd)[1]
    
    grad_jitted = jax.jit(jax.grad(weight_from_pos_quat, argnums=(0, 1,)))

    optimizer_pos = optax.adam(learning_rate=1e-4, b1=0.7)
    optimizer_quat = optax.adam(learning_rate=4e-3)

    @jax.jit
    def optimizer_kernel(st, i):
        opt_state_pos, opt_state_quat, pos, quat, observed_rgbd = st
        grad_pos, grad_quat = grad_jitted(pos, quat, observed_rgbd)
        updates_pos, opt_state_pos = optimizer_pos.update(-grad_pos, opt_state_pos)
        updates_quat, opt_state_quat = optimizer_quat.update(-grad_quat, opt_state_quat)
        pos = optax.apply_updates(pos, updates_pos)
        quat = optax.apply_updates(quat, updates_quat)
        return (opt_state_pos, opt_state_quat, pos, quat, observed_rgbd), (pos, quat)

    @jax.jit
    def unfold_300_steps(st):
        ret_st, _ = jax.lax.scan(optimizer_kernel, st, jnp.arange(300))
        return ret_st

    def get_initial_tracker_state(Xs_WP):
        opt_state_pos = optimizer_pos.init(Xs_WP._position)
        opt_state_quat = optimizer_quat.init(Xs_WP._quaternion)
        pos = Xs_WP._position
        quat = Xs_WP._quaternion
        tracker_state = (opt_state_pos, opt_state_quat, pos, quat, None)
        return tracker_state
    
    def update_tracker_state(tracker_state, new_observed_rgbd):
        updated_tracker_state = (*tracker_state[:-1], new_observed_rgbd)
        (opt_state_pos, opt_state_quat, pos, quat, _) = unfold_300_steps(updated_tracker_state)
        return (pos, quat), (opt_state_pos, opt_state_quat, pos, quat, new_observed_rgbd)
    
    return (get_initial_tracker_state, update_tracker_state)

def get_default_multiobject_model_for_patchtracking(renderer):
    depth_scale, color_scale, mindepth, maxdepth = 0.0001, 0.002, -20.0, 20.0
    model = b3d.patch_tracking.model.multiple_object_model_factory(
        renderer,
        b3d.likelihoods.get_uniform_multilaplace_image_dist_with_fixed_params(
            renderer.height, renderer.width, depth_scale, color_scale, mindepth, maxdepth
        ),
        r.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1)
    )

    return model