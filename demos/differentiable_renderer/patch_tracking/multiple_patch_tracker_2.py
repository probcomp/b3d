from b3d import Pose
import b3d
import rerun as rr
import numpy as np
from tqdm import tqdm
import b3d.patch_tracking.tracking as tracking
import demos.differentiable_renderer.patch_tracking.demo_utils as du

rr.init("multiple_patch_tracking_2")
rr.connect("127.0.0.1:8812")

width, height, fx, fy, cx, cy, near, far = 100, 128, 64.0, 64.0, 64.0, 64.0, 0.001, 16.0
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)

X_WC, rgbs, xyzs_W, observed_rgbds = du.get_rotating_box_data(renderer)

# Get the patch meshes
(patch_vertices_P, patch_faces, patch_vertex_colors, Xs_WP) = tracking.get_patches_with_default_centers(rgbs, xyzs_W, X_WC, fx)

# Get a patch tracker
model = tracking.get_default_multiobject_model_for_patchtracking(renderer)
(get_initial_tracker_state, update_tracker_state) = tracking.get_patch_tracker(
    model, patch_vertices_P, patch_faces, patch_vertex_colors, X_WC=Pose.identity()
)

tracker_state = get_initial_tracker_state(Xs_WP)

# Run the tracker over each frame
all_positions = []
all_quaternions = []
for timestep in tqdm(range(30)):
    (pos, quats), tracker_state = update_tracker_state(tracker_state, observed_rgbds[timestep])
    all_positions.append(pos)
    all_quaternions.append(quats)

# Visualize the result
for i in range(observed_rgbds.shape[0]):
    rr.set_time_sequence("frame--tracking", i)
    rr.log("/3D/tracked_points", rr.Points3D(
        positions = all_positions[i],
        radii=0.0075*np.ones(all_positions[i].shape[0]),
        colors=np.repeat(np.array([0,0,255])[None,...], all_positions[i].shape[0], axis=0))
    )