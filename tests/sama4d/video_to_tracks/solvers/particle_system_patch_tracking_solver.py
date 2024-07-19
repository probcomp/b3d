from tests.common.solver import Solver
import b3d
from b3d import Pose
import b3d.chisight.patch_tracking as tracking
import jax.numpy as jnp
import rerun as rr
import b3d.chisight.dense.differentiable_renderer as diffrend
import jax

class AdamPatchTracker_UsingSingleframeParticleSystemTraces(Solver):
    # Coordinate frames:
    # W - world
    # C - camera
    # P - patch (used to store mesh vertex positions in the patch frame)
    #
    # Note that the patch model trace we generate here is told to use the camera
    # frame as it's world frame; this function needs to handle
    # conversion from the trace's world frame (our camera frame)
    # to the task's world frame (our world frame).

    def __init__(self):
        self.get_trace = None
        self.all_positions_C = None
        self.all_quaternions_C = None
        self.all_positions_W = None
        self.all_quaternions_W = None
        self.patches = None
        self.poses_CP_at_time0 = None

    def solve(self, task_spec):
        video = task_spec["video"]
        assert video.shape[-1] == 4, "Expected video to be RGBD"
        poses_WC = task_spec["poses_WC"]
        initial_patch_centers_2D = task_spec["initial_keypoint_positions_2D"]
        r = task_spec["renderer"]
        fx, fy, cx, cy = r.fx, r.fy, r.cx, r.cy

        (patches, poses_CP_at_time0, _) = tracking.get_patches(
            initial_patch_centers_2D, video, Pose.identity(), fx, fy, cx, cy
        )
        self.patches = patches
        self.poses_CP_at_time0 = poses_CP_at_time0

        model = tracking.get_default_multiobject_model_for_patchtracking(r)
        (get_initial_tracker_state, update_tracker_state, get_trace) = tracking.get_adam_optimization_patch_tracker(
            model, patches
        )
        self.get_trace = get_trace
        tracker_state = get_initial_tracker_state(poses_CP_at_time0)
        self.all_positions_C = []
        self.all_quaternions_C = []
        self.all_positions_W = []
        self.all_quaternions_W = []
        observed_rgbds = video

        N_FRAMES = observed_rgbds.shape[0]
        for timestep in range(N_FRAMES):
            (pos_C, quats_C), tracker_state = update_tracker_state(tracker_state, observed_rgbds[timestep])
            pose_W = poses_WC[timestep] @ Pose(pos_C, quats_C)
            self.all_positions_W.append(pose_W.pos)
            self.all_quaternions_W.append(pose_W.xyzw)
            self.all_positions_C.append(pos_C)
            self.all_quaternions_C.append(quats_C)

        keypoints_3D_C = jnp.stack(self.all_positions_C)
        inferred_keypoints_2D = b3d.camera.screen_from_camera(
            keypoints_3D_C, r.get_intrinsics_object()
        )[:, :, ::-1]
        return inferred_keypoints_2D

    def visualize_solver_state(self, task_spec):
        pos0_C = self.poses_CP_at_time0.pos
        quat0_C = self.poses_CP_at_time0.xyzw
        trace = self.get_trace(pos0_C, quat0_C, task_spec["video"][0])

        _rerun_log_patchtracking_trace(
            trace, task_spec["renderer"], prefix="patch_tracking_initialization",
            static=True,
            # Viz uses World coordinate frame; trace uses Camera coordinate frame.
            # Hence, we need this transformation.
            transform_Viz_Trace=task_spec["poses_WC"][0]
        )

        for t in range(len(self.all_positions_W)):
            rr.set_time_sequence("frame", t)
            trace = self.get_trace(self.all_positions_C[t], self.all_quaternions_C[t], task_spec["video"][t])

            _rerun_log_patchtracking_trace(
                trace, task_spec["renderer"], prefix="PatchTrackingTrace",
                # Viz uses World coordinate frame; trace uses Camera coordinate frame.
                # Hence, we need this transformation.
                transform_Viz_Trace=task_spec["poses_WC"][t]
            )

def _rerun_log_patchtracking_trace(trace, renderer, prefix, transform_Viz_Trace, static=False):
    subtrace = trace.get_subtrace(("obs",))
    (observed_rgbd, metadata) = subtrace.get_retval()
    rr.log(f"/{prefix}/rgb/observed", rr.Image(observed_rgbd[:, :, :3]), static=static)
    rr.log(f"/{prefix}/depth/observed", rr.DepthImage(observed_rgbd[:, :, 3]), static=static)

    # Visualization path for the average render,
    # if the likelihood metadata contains the output of the differentiable renderer.
    if "diffrend_output" in metadata:
        weights, attributes = metadata["diffrend_output"]
        avg_obs = diffrend.dist_params_to_average(weights, attributes, jnp.zeros(4))
        avg_obs_rgb_clipped = jnp.clip(avg_obs[:, :, :3], 0, 1)
        avg_obs_depth_clipped = jnp.clip(avg_obs[:, :, 3], 0, 1)
        rr.log(f"/{prefix}/rgb/average_render", rr.Image(avg_obs_rgb_clipped), static=static)
        rr.log(f"/{prefix}/depth/average_render", rr.DepthImage(avg_obs_depth_clipped), static=static)

    # 3D:
    rr.log(f"/{prefix}/3D/", rr.Transform3D(translation=transform_Viz_Trace.pos, mat3x3=transform_Viz_Trace.rot.as_matrix()), static=static)

    pose_WC = trace.get_choices()["particle_dynamics", "state0", "initial_camera_pose"]
    poses_WO = trace.get_choices()("particle_dynamics")("state0")("object_poses").c.v

    mesh_C = subtrace.get_args()[0]
    mesh_W = b3d.Mesh(pose_WC.apply(mesh_C.vertices), mesh_C.faces, mesh_C.vertex_attributes)
    rr.log(f"/{prefix}/3D/mesh", rr.Mesh3D(
        vertex_positions=mesh_W.vertices.reshape(-1, 3),
        triangle_indices=mesh_W.faces,
        vertex_colors=mesh_W.vertex_attributes.reshape(-1, 3)
    ), static=static)

    rr.log(f"/{prefix}/3D/camera",
        rr.Pinhole(
            focal_length=[float(renderer.fx), float(renderer.fy)],
            width=renderer.width,
            height=renderer.height,
            principal_point=jnp.array([renderer.cx, renderer.cy]),
            ), static=static
        )

    rr.log(f"/{prefix}/3D/camera", rr.Transform3D(translation=pose_WC.pos, mat3x3=pose_WC.rot.as_matrix()), static=static)
    xyzs_C = b3d.utils.xyz_from_depth(observed_rgbd[:, :, 3], renderer.fx, renderer.fy, renderer.cx, renderer.cy)
    xyzs_W = pose_WC.apply(xyzs_C)
    rr.log(f"/{prefix}/3D/gt_pointcloud", rr.Points3D(
        positions=xyzs_W.reshape(-1,3),
        colors=observed_rgbd[:, :, :3].reshape(-1,3),
        radii = 0.001*jnp.ones(xyzs_W.reshape(-1,3).shape[0])),
        static=static
    )

    patch_centers_W = jax.vmap(lambda X_WO: X_WO.pos)(poses_WO)
    rr.log(
        f"/{prefix}/3D/patch_centers",
        rr.Points3D(positions=patch_centers_W, colors=jnp.array([0., 0., 1.]), radii=0.003),
        static=static
    )
