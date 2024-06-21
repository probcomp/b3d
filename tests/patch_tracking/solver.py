from tests.common.solver import Solver
from b3d.chisight.dense.model import rr_log_uniformpose_meshes_to_image_model_trace
import b3d.chisight.dense.patch_tracking as tracking
import jax.numpy as jnp
import rerun as rr

class AdamPatchTracker(Solver):
    def __init__(self):
        self.get_trace = None
        self.all_positions = None
        self.all_quaternions = None
        self.mesh = None
        self.Xs_WP_init = None
    
    def solve(self, task_spec):
        video = task_spec["video"]
        assert video.shape[-1] == 4, "Expected video to be RGBD"
        X_WC = task_spec["camera_pose"]
        initial_patch_centers_2D = task_spec["initial_patch_positions_2D"]
        r = task_spec["renderer"]
        fx, fy, cx, cy = r.fx, r.fy, r.cx, r.cy

        (patch_vertices_P, patch_faces, patch_vertex_colors, Xs_WP, _) = tracking.get_patches(
            initial_patch_centers_2D, video, X_WC, fx, fy, cx, cy
        )
        self.mesh = (patch_vertices_P, patch_faces, patch_vertex_colors)
        self.Xs_WP_init = Xs_WP

        model = tracking.get_default_multiobject_model_for_patchtracking(r)
        (get_initial_tracker_state, update_tracker_state, get_trace) = tracking.get_adam_optimization_patch_tracker(
            model, patch_vertices_P, patch_faces, patch_vertex_colors, X_WC=X_WC
        )
        self.get_trace = get_trace
        tracker_state = get_initial_tracker_state(Xs_WP)
        self.all_positions = []
        self.all_quaternions = []
        observed_rgbds = video

        N_FRAMES = observed_rgbds.shape[0]
        for timestep in range(N_FRAMES):
            (pos, quats), tracker_state = update_tracker_state(tracker_state, observed_rgbds[timestep])
            self.all_positions.append(pos)
            self.all_quaternions.append(quats)

        return jnp.stack(self.all_positions)
    
    def visualize_solver_state(self, task_spec):
        pos0 = self.Xs_WP_init.pos
        quat0 = self.Xs_WP_init.xyzw
        trace = self.get_trace(pos0, quat0, task_spec["video"][0])
        rr_log_uniformpose_meshes_to_image_model_trace(
            trace, task_spec["renderer"], prefix="patch_tracking_initialization",
            timeless=True
        )

        for t in range(len(self.all_positions)):
            rr.set_time_sequence("frame", t)
            trace = self.get_trace(self.all_positions[t], self.all_quaternions[t], task_spec["video"][t])
            rr_log_uniformpose_meshes_to_image_model_trace(trace, task_spec["renderer"], prefix="PatchTrackingTrace")