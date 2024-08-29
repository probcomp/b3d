import genjax
import jax
import jax.numpy as jnp
import optax
from genjax import ChoiceMapBuilder as C

import b3d
import b3d.chisight.dense.differentiable_renderer
import b3d.chisight.dense.differentiable_renderer as r
import b3d.chisight.dense.likelihoods as likelihoods
from b3d import Mesh, Pose
from b3d.chisight.particle_system import make_dense_gps_model


def get_patches(centers, rgbds, pose_WC, fx, fy, cx, cy):
    """
    Centers given as (N, 2) storing (x, y) pixel coordinates.
    """
    depths = rgbds[..., 3]
    xyzs_C = b3d.utils.xyz_from_depth_vectorized(depths, fx, fy, cx, cy)
    xyzs_W = pose_WC.apply(xyzs_C)
    return get_patches_from_pointcloud(centers, rgbds[..., :3], xyzs_W, pose_WC, fx)


def get_patches_from_pointcloud(centers, rgbs, xyzs_W, pose_WC, fx):
    """
    Centers given as (N, 2) storing (x, y) pixel coordinates.
    """
    xyzs_C = pose_WC.inv().apply(xyzs_W)

    # TODO: this would be better to do in terms of the min x dist and y dist
    # between any two centers
    pairwise_euclidean_dists = jnp.linalg.norm(
        centers[:, None] - centers[None], axis=-1
    )
    min_nonzero_dist = jnp.min(
        jnp.where(pairwise_euclidean_dists != 0, pairwise_euclidean_dists, jnp.inf)
    )
    del_pix = jnp.astype(jnp.ceil(min_nonzero_dist / (2 * jnp.sqrt(2))), int)
    del_pix = max(del_pix, 2)

    def get_patch(center):
        center = jnp.astype(jnp.round(center), int)
        center_x, center_y = center[0], center[1]
        patch_points_C = jax.lax.dynamic_slice(
            xyzs_C[0],
            (center_y - del_pix, center_x - del_pix, 0),
            (2 * del_pix - 1, 2 * del_pix - 1, 3),
        ).reshape(-1, 3)
        patch_rgbs = jax.lax.dynamic_slice(
            rgbs[0],
            (center_y - del_pix, center_x - del_pix, 0),
            (2 * del_pix - 1, 2 * del_pix - 1, 3),
        ).reshape(-1, 3)
        patch_vertices_C, patch_faces, patch_vertex_colors, _patch_face_colors = (
            b3d.make_mesh_from_point_cloud_and_resolution(
                patch_points_C, patch_rgbs, patch_points_C[:, 2] / fx * 2.0
            )
        )
        num_nonzero = jnp.sum(jnp.where(patch_points_C[..., 2] != 0, 1, 0))
        mean_position_nonzero = jnp.sum(patch_points_C, axis=0) / num_nonzero
        pose_CP = Pose.from_translation(mean_position_nonzero)
        pose_WP = pose_WC @ pose_CP
        patch_vertices_P = pose_CP.inv().apply(patch_vertices_C)
        patches = Mesh(patch_vertices_P, patch_faces, patch_vertex_colors)
        return (patches, pose_WP, patch_points_C)

    return jax.vmap(get_patch, in_axes=(0,))(centers)


def get_adam_optimization_patch_tracker(model, patches, pose_WC=Pose.identity()):
    """
    Args:
        - model: instance of the multiple object model from b3d.patch_tracking.model
        - patch_vertices_P: The vertices of the patch in the patch's local frame. Shape (N, V, 3)
        - patch_faces: The faces of the patch. Shape (N, F, 3)
        - patch_vertex_colors: The vertex colors of the patch. Shape (N, V, 3)
        - pose_WC: The camera pose. Default is the identity pose.

    Returns:
    - get_initial_tracker_state:
        A function from the initial patch poses, poses_WP, to an initial state object `tracker_state` for the patch tracker.
    - update_tracker_state:
        A function from `(tracker_state, new_observed_rgbd)` to a tuple `(new_patch_poses, updated_tracker_state)`,
        where `new_patch_poses` is a pair (positions, quaternions) for the updated patch poses,
        and `new_tracker_state` is the updated tracker state.
        At time 0, `new_observed_rgbd` should be the observed_rgbd for the frame at t=0.
    - get_trace: A function s.t. `get_trace(pos, quat, observed_rgbd)` returns a trace for `model` with the patches
        at the given positions and quaternions.
    """

    def allidx_chm(x):
        return genjax.ChoiceMap.idx(jnp.arange(x.shape[0], dtype=int), x)

    @jax.jit
    def importance_from_pos_quat(positions, quaternions, observed_rgbd):
        key = jax.random.PRNGKey(
            0
        )  # This value shouldn't matter, in the current model version.

        max_num_timesteps = genjax.Pytree.const(1)
        num_particles = genjax.Pytree.const(positions.shape[0])
        num_clusters = genjax.Pytree.const(positions.shape[0])
        relative_particle_poses_prior_params = (Pose.identity(), 0.5, 0.25)
        initial_object_poses_prior_params = (Pose.identity(), 2.0, 0.5)
        camera_pose_prior_params = (Pose.identity(), 0.1, 0.1)

        model_args = (
            (
                max_num_timesteps,  # const object
                num_particles,  # const object
                num_clusters,  # const object
                relative_particle_poses_prior_params,
                initial_object_poses_prior_params,
                camera_pose_prior_params,
            ),
            (patches, ()),
        )

        particle_poses = jax.tree.map(
            lambda arr: jnp.tile(
                arr, (genjax.Pytree.tree_unwrap_const(num_particles), 1)
            ),
            Pose.identity(),
        )
        object_assignments = jnp.arange(
            genjax.Pytree.tree_unwrap_const(num_particles), dtype=int
        )
        object_poses = jax.vmap(
            lambda pos, quat: Pose.from_vec(jnp.concatenate([pos, quat])),
            in_axes=(0, 0),
        )(positions, quaternions)
        vis_mask = jnp.ones(
            (genjax.Pytree.tree_unwrap_const(num_particles),), dtype=int
        )

        constraints = C.d(
            {
                "particle_dynamics": C["state0"].set(
                    C.d(
                        {
                            "particle_poses": allidx_chm(particle_poses),
                            "object_assignments": allidx_chm(object_assignments),
                            "object_poses": allidx_chm(object_poses),
                            "initial_camera_pose": pose_WC,
                            "initial_visibility": allidx_chm(vis_mask),
                        }
                    )
                ),
                "obs": C["image"].set(observed_rgbd),
            }
        )

        trace, weight = model.importance(key, constraints, model_args)
        return trace, weight

    def weight_from_pos_quat(pos, quat, observed_rgbd):
        return importance_from_pos_quat(pos, quat, observed_rgbd)[1]

    @jax.jit
    def get_trace(pos, quat, observed_rgbd):
        return importance_from_pos_quat(pos, quat, observed_rgbd)[0]

    grad_jitted = jax.jit(
        jax.grad(
            weight_from_pos_quat,
            argnums=(
                0,
                1,
            ),
        )
    )

    optimizer_pos = optax.adam(learning_rate=1e-4, b1=0.7)
    optimizer_quat = optax.adam(learning_rate=4e-3)

    @jax.jit
    def optimizer_kernel(st, i):
        opt_state_pos, opt_state_quat, pos, quat, observed_rgbd = st
        # og_pos, og_quat = pos, quat
        # weight = weight_from_pos_quat(pos, quat, observed_rgbd)
        grad_pos, grad_quat = grad_jitted(pos, quat, observed_rgbd)
        updates_pos, opt_state_pos = optimizer_pos.update(-grad_pos, opt_state_pos)
        updates_quat, opt_state_quat = optimizer_quat.update(-grad_quat, opt_state_quat)
        pos = optax.apply_updates(pos, updates_pos)
        quat = optax.apply_updates(quat, updates_quat)
        # jax.debug.print("Weight: {x}", x=weight)
        # jax.debug.print("Pos grad magnitude: {x}", x=jnp.linalg.norm(grad_pos))
        # jax.debug.print("Quat grad magnitude: {x}", x=jnp.linalg.norm(grad_quat))
        # jax.debug.print("Position change: {x}", x=jnp.linalg.norm(og_pos - pos))
        # jax.debug.print("Quaternion change: {x}", x=jnp.linalg.norm(og_quat - quat))
        return (opt_state_pos, opt_state_quat, pos, quat, observed_rgbd), (pos, quat)

    @jax.jit
    def unfold_300_steps(st):
        ret_st, _ = jax.lax.scan(optimizer_kernel, st, jnp.arange(300))
        return ret_st

    def get_initial_tracker_state(poses_WP):
        opt_state_pos = optimizer_pos.init(poses_WP._position)
        opt_state_quat = optimizer_quat.init(poses_WP._quaternion)
        pos = poses_WP._position
        quat = poses_WP._quaternion
        tracker_state = (opt_state_pos, opt_state_quat, pos, quat, None)
        return tracker_state

    def update_tracker_state(tracker_state, new_observed_rgbd):
        updated_tracker_state = (*tracker_state[:4], new_observed_rgbd)
        (opt_state_pos, opt_state_quat, pos, quat, _) = unfold_300_steps(
            updated_tracker_state
        )
        return (pos, quat), (
            opt_state_pos,
            opt_state_quat,
            pos,
            quat,
            new_observed_rgbd,
        )

    return (get_initial_tracker_state, update_tracker_state, get_trace)


def get_default_multiobject_model_for_patchtracking(renderer):
    depth_scale, color_scale, mindepth, maxdepth = 0.0001, 0.002, -20.0, 20.0
    likelihood = likelihoods.get_uniform_multilaplace_image_dist_with_fixed_params(
        renderer.height, renderer.width, depth_scale, color_scale, mindepth, maxdepth
    )

    @genjax.gen
    def wrapped_likelihood(mesh: b3d.Mesh, args):
        weights, attributes = (
            b3d.chisight.dense.differentiable_renderer.render_to_rgbd_dist_params(
                renderer,
                mesh.vertices,
                mesh.faces,
                mesh.vertex_attributes,
                r.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1),
            )
        )
        obs = likelihood(weights, attributes) @ "image"
        return obs, {"diffrend_output": (weights, attributes)}

    model = make_dense_gps_model(wrapped_likelihood)

    return model
