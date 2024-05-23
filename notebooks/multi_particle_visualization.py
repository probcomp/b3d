import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
#from b3d.utils import unproject_depth
import rerun as rr
import genjax
from tqdm import tqdm

rr.init("demo.py")
rr.connect("127.0.0.1:8812")
rr.save('multi_particle_visualization.rrd')

# set up renderer
width=128
height=128
fx=64.0
fy=64.0
cx=64.0
cy=64.0
near=0.001
far=16.0

class Intrinsics:
    def __init__(self, width=128, height=128, fx=64., fy=64., cx=64., cy=64., near=0.001, far=16.):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.near = near
        self.far = far
    # width=128
    # height=128
    # fx=64.0
    # fy=64.0
    # cx=64.0
    # cy=64.0
    # near=0.001
    # far=16.0

intrinsics = Intrinsics(
    width=128,
    height=128,
    fx=64.0,
    fy=64.0,
    cx=64.0,
    cy=64.0,
    near=0.001,
    far=16.0
)

renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)


# Define pose math libraries
def rotation_from_axis_angle(axis, angle):
    """Creates a rotation matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The rotation matrix. Shape (3, 3)
    """
    sina = jnp.sin(angle)
    cosa = jnp.cos(angle)
    direction = axis / jnp.linalg.norm(axis)
    # rotation matrix around unit vector
    R = jnp.diag(jnp.array([cosa, cosa, cosa]))
    R = R + jnp.outer(direction, direction) * (1.0 - cosa)
    direction = direction * sina
    R = R + jnp.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return R

def transform_from_rot(rotation):
    """Creates a pose matrix from a rotation matrix.

    Args:
        rotation (jnp.ndarray): The rotation matrix. Shape (3, 3)
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return jnp.vstack(
        [jnp.hstack([rotation, jnp.zeros((3, 1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_axis_angle(axis, angle):
    """Creates a pose matrix from an axis and angle.

    Args:
        axis (jnp.ndarray): The axis vector. Shape (3,)
        angle (float): The angle in radians.
    Returns:
        jnp.ndarray: The pose matrix. Shape (4, 4)
    """
    return transform_from_rot(rotation_from_axis_angle(axis, angle))


def unproject_depth(depth, intrinsics):
    """Unprojects a depth image into a point cloud.

    Args:
        depth (jnp.ndarray): The depth image. Shape (H, W)
        intrinsics (b.camera.Intrinsics): The camera intrinsics.
    Returns:
        jnp.ndarray: The point cloud. Shape (H, W, 3)
    """
    mask = (depth < intrinsics.far) * (depth > intrinsics.near)
    depth = depth * mask + intrinsics.far * (1.0 - mask)
    y, x = jnp.mgrid[: depth.shape[0], : depth.shape[1]]
    x = (x - intrinsics.cx) / intrinsics.fx
    y = (y - intrinsics.cy) / intrinsics.fy
    point_cloud_image = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * depth[:, :, None]
    return point_cloud_image

unproject_depth_vec = jax.vmap(unproject_depth, (0, None))

# calculate sequence of pose transformations
r_mat = transform_from_axis_angle(jnp.array([0,0,1]), jnp.pi/2)
vec_transform_axis_angle = jax.vmap(transform_from_axis_angle, (None, 0))
rots = vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))



# Set up data
mesh_path = os.path.join(b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
mesh = trimesh.load(mesh_path)

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)

cam_inv_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.15, 0.15, 0.0]),
    jnp.array([0.0, 0.0, 0.0])
).inv()


in_place_rots = b3d.Pose.from_matrix(rots)


compound_pose = cam_inv_pose @ in_place_rots #in_place_rot

with open("poses.npy", "wb") as f:
    jnp.savez(f,
              object_positions=compound_pose.pos,
              object_quaternions=compound_pose.quat
              )
    
with open("poses.npy", "rb") as f:
    data = np.load(f)
    object_positions = data["object_positions"]
    object_quaternions = data["object_quaternions"]

rgbs, depths = renderer.render_attribute_many(
    compound_pose[:,None,...],
    object_library.vertices,
    object_library.faces,
    jnp.array([[0, len(object_library.faces)]]),
    object_library.attributes
)

xyzs = unproject_depth_vec(depths, intrinsics)

# Set up generative model 

num_layers = 2048
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far, num_layers)
model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (jnp.float32(30.0), jnp.float32(0.01))
# TODO: explain
inlier_score, outlier_prob = (jnp.float32(50.0), jnp.float32(0.001))
# TODO: explain
color_multiplier, depth_multiplier = (jnp.float32(3000.0), jnp.float32(3000.0))


# Defines the enumeration schedule.
key = jax.random.PRNGKey(0)
# Gridding on translation only.

delta = 0.01
translation_deltas = Pose.concatenate_poses([jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-delta, delta, 11),
        jnp.linspace(-delta, delta, 11),
        jnp.linspace(-delta, delta, 11),
    ),
    axis=-1,
).reshape(-1, 3)), Pose.identity()[None,...]])
# Sample orientations from a VMF to define a "grid" over orientations.
rotation_deltas = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 11*11*11),
    Pose.identity(),
    0.00001, 1000.0
), Pose.identity()[None,...]])
all_deltas =  Pose.stack_poses([translation_deltas, rotation_deltas])


# patch-wise trajectory tracker
def get_trajs(key, center_1, center_2, del_pix=5):
    # Make empty library
    object_library = b3d.MeshLibrary.make_empty_library()

    local_points = xyzs[0,center_1-del_pix:center_1+del_pix,center_2-del_pix:center_2+del_pix,:].reshape(-1,3)
    local_rgbs = rgbs[0,center_1-del_pix:center_1+del_pix,center_2-del_pix:center_2+del_pix,:].reshape(-1,3)
    patch_center = xyzs[0,center_1,center_2,:]

    point_cloud = local_points
    point_cloud_colors = local_rgbs

    # Create new mesh.
    vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud, point_cloud_colors, point_cloud[:,2] / fx * 2.0
    )
    object_pose = Pose.from_translation(vertices.mean(0))
    vertices = object_pose.inverse().apply(vertices)
    object_library.add_object(vertices, faces, vertex_colors)
    model_args = b3d.model.ModelArgs(color_error, depth_error,
                                inlier_score, outlier_prob,
                                color_multiplier, depth_multiplier)

    trace, _ = importance_jit(
        jax.random.PRNGKey(0),
        genjax.choice_map(
            dict([
                ("camera_pose", Pose.identity()),
                ("object_pose_0", object_pose),
                ("object_0", 0),
                ("observed_rgb_depth", (rgbs[0], xyzs[0,...,2])),
            ])
        ),
        (jnp.arange(1),model_args, object_library)
    )


    FINAL_T = len(xyzs)
    patches = []
    patch_centers = []

    for T_observed_image in tqdm(range(0, FINAL_T)):
        # Constrain on new RGB and Depth data.
        trace = b3d.update_choices_jit(trace, key,
            genjax.Pytree.const(["observed_rgb_depth"]),
            (rgbs[T_observed_image],xyzs[T_observed_image,...,2])
        )
        trace,key = b3d.enumerate_and_select_best_move(trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
        trace,key = b3d.enumerate_and_select_best_move(trace, genjax.Pytree.const([f"object_pose_0"]), key, all_deltas)

        patch_center = (trace["camera_pose"].inv() @ trace[f"object_pose_0"]).apply(jnp.mean(object_library.vertices,axis=0))
        patch = (trace["camera_pose"].inv() @ trace[f"object_pose_0"]).apply(object_library.vertices)
        patches.append(patch)
        patch_centers.append(patch_center)

    return jnp.array(patches), jnp.array(patch_centers)


# Generate surface patch tracks
center_arr = []
for i in np.arange(35,100,5):
    row_arr = []
    for j in np.arange(45,90,5):
        row_arr.append(get_trajs(key, i, j)[1])
    center_arr.append(row_arr)

center_arr = jnp.array(center_arr)
center_arr = center_arr.reshape(-1,len(xyzs),3)

# Visualize surface patch tracks
num_points = center_arr.shape[0]

t = 0
for t in range(len(xyzs)):
    rr.set_time_sequence("frame", t)

    points = rr.Points3D(positions=xyzs[t].reshape(-1,3), colors=rgbs[t].reshape(-1,3), radii = 0.0005*np.ones(xyzs[t].reshape(-1,3).shape[0]))
    rr.log("cloud1", points)

    points2 = rr.Points3D(center_arr[:,t,:], radii=0.0075*np.ones(center_arr.shape[0]), colors =np.repeat(np.array([0,0,255])[None,...], num_points, axis=0))
    rr.log("cloud2", points2)

    num_frames_trail = 5
    if t > num_frames_trail:
        rr.log(
            "strips",
            rr.LineStrips3D(np.array(center_arr[:,t:t-num_frames_trail:-1,...]), colors=np.repeat(np.array([0,255,0])[None,...], num_points, axis=0), 
            radii= 0.0025*np.ones(num_points))
        )

with open("tracks.npy", "wb") as f:
    jnp.savez(f, tracks=center_arr, pointcloud_xyzs=xyzs, pointcloud_rgbs=rgbs)

with open("tracks.npy", "rb") as f:
    data = np.load(f)
    tracks = data["tracks"]
    pointcloud_xyzs = data["pointcloud_xyzs"]
    pointcloud_rgbs = data["pointcloud_rgbs"]