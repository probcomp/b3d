import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt 
from carvekit.api.high import HiInterface


###################################
# Setup
###################################

PORT = 8813
rr.init("SHOUT")
rr.connect(addr=f'127.0.0.1:{PORT}')

## get data (43*5 to end (228*5) of original full video)
subsampling_frame = 5
path = os.path.join(b3d.get_assets(), "shared_data_bucket/input_data/demo_reel_place_and_pickup.r3d.video_input_shout_to_end.npz")
video_input = b3d.VideoInput.load(path) 

image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy,cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

rgbs_all = video_input.rgb / 255.0
xyzs = video_input.xyz[::subsampling_frame, ...]
rgbs = rgbs_all[::subsampling_frame, ...]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs, (xyzs.shape[1], xyzs.shape[2], 3), "linear"
), 0.0, 1.0)


###################################
# Relevant time markers
###################################
BG_MESH_T = 0  # when to acquire background mesh
FULL_SHOUT_T = 32  # first frame where full shout object is shown, no hand
END_T = rgbs_resized.shape[0]

###################################
# Background mesh acquisition
###################################
point_cloud_for_mesh = xyzs[BG_MESH_T].reshape(-1, 3)
colors_for_mesh = rgbs_resized[BG_MESH_T].reshape(-1, 3)

_bg_vertices, bg_faces, bg_vertex_colors, bg_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud_for_mesh, 
    colors_for_mesh, 
    point_cloud_for_mesh[:,2] / fx 
)

bg_object_pose = Pose.from_translation(_bg_vertices.mean(0))  # (approximate) CAMERA frame object pose
bg_vertices = bg_object_pose.inverse().apply(_bg_vertices)  # (approximate) WORLD frame vertices


## Build a subset mesh of the background (for faster enumeration)
subset1 = ((point_cloud_for_mesh < point_cloud_for_mesh.mean(0) * 1.5)).all(1)
subset2 = ((point_cloud_for_mesh > point_cloud_for_mesh.mean(0) * 0.75)).all(1)
subset = subset1 | subset2 

point_cloud_subset = point_cloud_for_mesh[subset]
colors_for_mesh_subset = colors_for_mesh[subset]


_bg_vertices_subset, bg_faces_subset, bg_vertex_colors_subset, bg_face_colors_subset = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud_subset, colors_for_mesh_subset, point_cloud_subset[:,2] / fx )


bg_object_pose_subset = Pose.from_translation(_bg_vertices_subset.mean(0))  # CAMERA frame object pose
bg_vertices_subset_unaligned = bg_object_pose_subset.inverse().apply(_bg_vertices_subset)  # WORLD frame vertices

# align the subset points to same world frame coords as whole bg points
vertices_subset_to_whole = bg_object_pose.as_matrix() @ jnp.linalg.inv(bg_object_pose_subset.as_matrix())
vertices_subset_to_whole_xfm = Pose(position=vertices_subset_to_whole[:3,-1], 
                                    quaternion=Pose.identity_quaternion)
bg_vertices_subset = vertices_subset_to_whole_xfm.apply(bg_vertices_subset_unaligned)


###################################
# Camera pose track
###################################


###################################
# SHOUT mesh acquisition
###################################


###################################
# Camera pose + Object pose track
###################################



from IPython import embed; embed()
