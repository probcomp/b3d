import numpy as np
import rerun as rr
from tqdm import tqdm

import b3d
import b3d.chisight.dense.patch_tracking as tracking
import demos.differentiable_renderer.patch_tracking.demo_utils as du
from b3d.chisight.dense.model import rr_log_uniformpose_meshes_to_image_model_trace

rr.init("multiple_patch_tracking_2")
rr.connect("127.0.0.1:8812")

width, height, fx, fy, cx, cy, near, far = 100, 128, 64.0, 64.0, 64.0, 64.0, 0.001, 16.0
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)

X_WC, rgbs, xyzs_W, observed_rgbds = du.get_rotating_box_data(renderer)

# Get the patch meshes
(patch_vertices_P, patch_faces, patch_vertex_colors, Xs_WP) = (
    tracking.get_patches_with_default_centers(rgbs, xyzs_W, X_WC, fx)
)

# Get a patch tracker
model = tracking.get_default_multiobject_model_for_patchtracking(renderer)

(get_initial_tracker_state, update_tracker_state, get_trace) = (
    tracking.get_adam_optimization_patch_tracker(
        model, patch_vertices_P, patch_faces, patch_vertex_colors, X_WC=X_WC
    )
)

tracker_state = get_initial_tracker_state(Xs_WP)

# Run the tracker over each frame
all_positions = []
all_quaternions = []

N_FRAMES = 6  # observed_rgbds.shape[0]
for timestep in tqdm(range(N_FRAMES)):
    (pos, quats), tracker_state = update_tracker_state(
        tracker_state, observed_rgbds[timestep]
    )
    all_positions.append(pos)
    all_quaternions.append(quats)

# Visualize the result
for i in range(N_FRAMES):
    rr.set_time_sequence("frame--tracking", i)

    # Log the inferred dense model trace from this timestep
    trace = get_trace(all_positions[i], all_quaternions[i], observed_rgbds[i])
    rr_log_uniformpose_meshes_to_image_model_trace(trace, renderer)

    rr.log(
        "/3D/tracked_points",
        rr.Points3D(
            positions=all_positions[i],
            radii=0.0075 * np.ones(all_positions[i].shape[0]),
            colors=np.repeat(
                np.array([0, 0, 255])[None, ...], all_positions[i].shape[0], axis=0
            ),
        ),
    )
