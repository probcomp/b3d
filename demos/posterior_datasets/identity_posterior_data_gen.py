import b3d
import numpy as np
import trimesh
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os

width = 200
height = 200
fx = 300.0
fy = 300.0
cx = 100.0
cy = 100.0
near = 0.001
far = 16.0

renderer = b3d.Renderer(int(width), int(height), fx, fy, cx, cy, near, far)

## rerun for mesh viz
import rerun as rr
PORT = 8812
rr.init("fork-knife")
rr.connect(addr=f"127.0.0.1:{PORT}")


## make meshes
### occluder mesh
object_library = b3d.MeshLibrary.make_empty_library()
occluder = trimesh.creation.box(extents=np.array([0.15, 0.1, 0.02]))
occluder_colors = jnp.tile(jnp.array([0.8, 0.8, 0.8])[None,...], (occluder.vertices.shape[0], 1))

occ_id = 0 # id in library
object_library.add_object(occluder.vertices, occluder.faces, attributes=occluder_colors)

# add fork and knife
fork_mesh_path = os.path.join(b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/030_fork/textured.obj")
knife_mesh_path = os.path.join(b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/032_knife/textured.obj")

for obj_mesh_path in [fork_mesh_path, knife_mesh_path]:
    obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
    object_library.add_trimesh(obj_mesh)

print(f"{object_library.get_num_objects()} objects in library")

## position the scene
camera_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.0, 0.2, 0.8]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
)

occluder_pose = b3d.Pose.from_pos(jnp.array([0.0, 0.05, 0.3]))


####
# Generate poses
####
NUM_IMAGES = 10

contact_parameters_to_pose = lambda cp: b3d.Pose(
    jnp.array([cp[0], 0.0, cp[1]]),  # fixed height (y) at 0 for table
    b3d.Rot.from_rotvec(jnp.array([0.0, cp[2], 0.0])).as_quat(),
)

get_scene_poses = lambda obj_cp: b3d.Pose.stack_poses([
    occluder_pose, contact_parameters_to_pose(obj_cp)
])

w = 0.2
cps = jax.random.uniform(jax.random.PRNGKey(110), (NUM_IMAGES, 3),
                         minval=jnp.array([-w,-w,0]),
                         maxval=jnp.array([w, w, jnp.pi]))
scene_poses = jax.vmap(get_scene_poses)(cps)
scene_poses_in_camera = camera_pose.inv() @ scene_poses

####
# viz meshes in rerun for sanity check  (e.g. make sure the objects aren't poking through the tabletop)
####


rr.log(
    f"/3d/mesh/occ",
    rr.Mesh3D(
        vertex_positions=scene_poses_in_camera[0][0].apply(object_library.vertices),
        indices=object_library.faces[object_library.ranges[0,0]: object_library.ranges[0,:].sum()],
        vertex_colors=object_library.attributes
    )
)
rr.log(
    f"/3d/mesh/obj",
    rr.Mesh3D(
        vertex_positions=scene_poses_in_camera[0][2].apply(object_library.vertices),
        indices=object_library.faces[object_library.ranges[1,0]: object_library.ranges[1,:].sum()],
        vertex_colors=object_library.attributes
    )
)



######
# render scenes
#####

## render scene with fork/knife
rgbs_fork, depths_fork = renderer.render_attribute_many(
                                            scene_poses_in_camera,
                                            object_library.vertices,
                                            object_library.faces,
                                            object_library.ranges[jnp.array([0,1])],
                                            object_library.attributes
                                        )

rgbs_knife, depths_knife = renderer.render_attribute_many(scene_poses_in_camera,
                                                          object_library.vertices,
                                                          object_library.faces,
                                                        object_library.ranges[jnp.array([0,2])],
                                                        object_library.attributes)

rr.log("fork", rr.Image(rgbs_fork[0]))
rr.log("knife", rr.Image(rgbs_knife[0]))

data_fork = b3d.io.VideoInput(
    rgb=(rgbs_fork * 255.0).astype(jnp.uint8),
    xyz=jax.vmap(b3d.xyz_from_depth,in_axes=(0,None, None, None, None))(depths_fork, fx, fy, cx, cy),
    camera_positions=jnp.array([camera_pose.pos for _ in range(NUM_IMAGES+1)]),
    camera_quaternions=jnp.array([camera_pose.quat for _ in range(NUM_IMAGES+1)]),
    camera_intrinsics_rgb=jnp.array([width, height, fx, fy, cx, cy, near, far]),
    camera_intrinsics_depth=jnp.array([width, height, fx, fy, cx, cy, near, far])
)
data_knife = b3d.io.VideoInput(
    rgb=(rgbs_knife * 255.0).astype(jnp.uint8),
    xyz=jax.vmap(b3d.xyz_from_depth,in_axes=(0,None, None, None, None))(depths_knife, fx, fy, cx, cy),
    camera_positions=jnp.array([camera_pose.pos for _ in range(NUM_IMAGES)]),
    camera_quaternions=jnp.array([camera_pose.quat for _ in range(NUM_IMAGES)]),
    camera_intrinsics_rgb=jnp.array([width, height, fx, fy, cx, cy, near, far]),
    camera_intrinsics_depth=jnp.array([width, height, fx, fy, cx, cy, near, far])
)

data_fork.save(os.path.join(b3d.get_root_path(),
                            "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_fork.npz"))
data_knife.save(os.path.join(b3d.get_root_path(),
                            "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_knife.npz"))
