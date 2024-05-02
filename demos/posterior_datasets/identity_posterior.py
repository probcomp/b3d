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
scale = 2.0
width, height, fx, fy, cx, cy, near, far = np.array([width, height, fx, fy, cx, cy, near, far]) * scale

renderer = b3d.Renderer(int(width), int(height), fx, fy, cx, cy, near, far) 

## rerun for mesh viz
import rerun as rr
PORT = 8812
rr.init("fork-knife")
rr.connect(addr=f"127.0.0.1:{PORT}")


## make meshes
### occluder mesh
object_library = b3d.MeshLibrary.make_empty_library()
occ_height = 1.0
occluder = trimesh.creation.box(extents=np.array([0.5, occ_height, 0.01]))
occluder_colors = np.ones((occluder.vertices.shape[0], 3)) * 0.5

occ_id = 0 # id in library
object_library.add_object(occluder.vertices, occluder.faces, attributes=occluder_colors)

# table mesh
table_thickness = 0.01
table = trimesh.creation.box(extents=np.array([20.0, table_thickness, 20.0]))
table_colors = np.ones((table.vertices.shape[0], 3)) 
table_id = 1 # id in library
object_library.add_object(table.vertices, table.faces, attributes=table_colors)

# add fork and knife
fork_mesh_path = os.path.join(b3d.get_root_path(), "assets/ycb_video_models/models/030_fork/textured.obj")
knife_mesh_path = os.path.join(b3d.get_root_path(), "assets/ycb_video_models/models/032_knife/textured.obj")

for obj_mesh_path in [fork_mesh_path, knife_mesh_path]:
    obj_mesh = trimesh.load(obj_mesh_path, force='mesh')

    obj_vertices = jnp.array(obj_mesh.vertices) * 5
    obj_vertices = obj_vertices - obj_vertices.mean(0)
    obj_faces = jnp.array(obj_mesh.faces)
    obj_vertex_colors = jnp.array(obj_mesh.visual.to_color().vertex_colors)[...,:3] / 255.0

    object_library.add_object(obj_vertices, obj_faces, attributes=obj_vertex_colors)

print(f"{object_library.get_num_objects()} objects in library")

## position the scene
camera_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.0, 1.0, 4.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
)

occluder_pose = b3d.Pose.from_pos(jnp.array([0.0, occ_height/2, 1.5]))
table_pose = b3d.Pose.from_pos(jnp.array([0.0, -0.5, 0.0]))


####
# Generate poses
####
NUM_IMAGES = 10

contact_parameters_to_pose = lambda cp: b3d.Pose(
    jnp.array([cp[0], 0.0, cp[1]]),  # fixed height (y) at 0 for table 
    b3d.Rot.from_rotvec(jnp.array([0.0, cp[2], 0.0])).as_quat(),
)

get_scene_poses = lambda obj_cp: b3d.Pose.stack_poses([
    occluder_pose, table_pose, contact_parameters_to_pose(obj_cp)
    ])

w = 1
cps = jax.random.uniform(jax.random.PRNGKey(0), (NUM_IMAGES, 3), 
                         minval=jnp.array([-w,-w,0]), 
                         maxval=jnp.array([w, w, jnp.pi]))
scene_poses = jax.vmap(get_scene_poses)(cps)
scene_poses_in_camera = camera_pose.inv() @ scene_poses

####
# viz meshes in rerun for sanity check  (e.g. make sure the objects aren't poking through the tabletop)
####

rr.log(
    f"/3d/mesh/table",
    rr.Mesh3D(
        vertex_positions=scene_poses_in_camera[0][0].apply(table.vertices),
        indices=table.faces,
        vertex_colors=table_colors
    )
)
rr.log(
    f"/3d/mesh/occ",
    rr.Mesh3D(
        vertex_positions=scene_poses_in_camera[0][1].apply(occluder.vertices),
        indices=occluder.faces,
        vertex_colors=occluder_colors
    )
)
rr.log(
    f"/3d/mesh/obj",
    rr.Mesh3D(
        vertex_positions=scene_poses_in_camera[0][2].apply(obj_vertices),
        indices=obj_faces,
        vertex_colors=obj_vertex_colors
    )
)


######
# render scenes
#####

### Render just empty scene
rgb_scene, depth_scene = renderer.render_attribute(
                                            scene_poses_in_camera[0][:2], 
                                            object_library.vertices,
                                            object_library.faces, 
                                            object_library.ranges[jnp.array([0,1])], 
                                            object_library.attributes
                                        )
plt.imshow(rgb_scene)

## render scene with fork/knife
rgbs_fork, depths_fork = renderer.render_attribute_many(
                                            scene_poses_in_camera, 
                                            object_library.vertices,
                                            object_library.faces, 
                                            object_library.ranges[jnp.array([0,1,2])], 
                                            object_library.attributes
                                        )

rgbs_knife, depths_knife = renderer.render_attribute_many(scene_poses_in_camera, 
                                                          object_library.vertices, 
                                                          object_library.faces, 
                                                        object_library.ranges[jnp.array([0,1,3])], 
                                                        object_library.attributes)
for i, rgb in enumerate(rgbs_knife):
    print(scene_poses_in_camera[i].pos[-1])
    plt.imshow(rgb)
    plt.show()

## viz
# for i, rgb in enumerate(rgbs_fork):
#     print(scene_poses_in_camera[i].pos[-1])
#     plt.imshow(rgb)
#     plt.show()
# plt.imshow(jnp.concatenate([rgbs_fork[0], rgbs_fork[1]], axis=1))


######
# save data. include a first frame that contains no object
######

data_fork = b3d.utils.VideoInput(
    rgb=jnp.concatenate([rgb_scene[None,...], rgbs_fork]), 
    xyz=jnp.concatenate([depth_scene[None,...], depths_fork]),
    camera_positions=jnp.array([camera_pose.pos for _ in range(NUM_IMAGES+1)]),
    camera_quaternions=jnp.array([camera_pose.quat for _ in range(NUM_IMAGES+1)]),
    camera_intrinsics_rgb=jnp.array([width, height, fx, fy, cx, cy, near, far])/scale,
    camera_intrinsics_depth=jnp.array([width, height, fx, fy, cx, cy, near, far])/scale
)
data_knife = b3d.utils.VideoInput(
    rgb=jnp.concatenate([rgb_scene[None,...], rgbs_knife]), 
    xyz=jnp.concatenate([depth_scene[None,...], depths_knife]),
    camera_positions=jnp.array([camera_pose.pos for _ in range(NUM_IMAGES)]),
    camera_quaternions=jnp.array([camera_pose.quat for _ in range(NUM_IMAGES)]),
    camera_intrinsics_rgb=jnp.array([width, height, fx, fy, cx, cy, near, far])/scale,
    camera_intrinsics_depth=jnp.array([width, height, fx, fy, cx, cy, near, far])/scale
)

data_fork.save(os.path.join(b3d.get_root_path(), 
                            "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_fork.npz"))
data_knife.save(os.path.join(b3d.get_root_path(), 
                            "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_knife.npz"))