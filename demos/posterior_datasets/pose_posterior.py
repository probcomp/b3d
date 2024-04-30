import b3d
import trimesh
import jax.numpy as jnp
import jax
import rerun as rr
import os 

rr.init("demo")
rr.connect("127.0.0.1:8812")

width = 200
height = 200
fx = 300.0
fy = 300.0
cx = 100.0
cy = 100.0
near = 0.001
far = 16.0
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)


object_library = b3d.MeshLibrary.make_empty_library()
mesh_path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured_simple.obj",
)
object_library.add_trimesh(trimesh.load(mesh_path))

camera_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.5, 0.0, 0.2]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
)

NUM_IMAGES = 100
w = 0.02
cps = jax.random.uniform(jax.random.PRNGKey(0), (NUM_IMAGES, 3), minval=jnp.array([-w, -w, -jnp.pi]), maxval=jnp.array([w, w, jnp.pi]))
object_poses = jax.vmap(b3d.contact_parameters_to_pose)(cps)

object_poses_in_cam_frame = camera_pose.inv() @ object_poses

rgb, depth = renderer.render_attribute_many(object_poses_in_cam_frame[:,None,...], object_library.vertices, object_library.faces, jnp.array([[0, len(object_library.faces)]]), object_library.attributes)
rgb = jnp.clip(rgb, 0.0, 1.0)

for i in range(NUM_IMAGES):
    rr.set_time_sequence("frame", i)
    rr.log("img", rr.Image(rgb[i]))

jnp.savez(b3d.get_root_path() / "assets/shared_data_bucket/datasets/posterior_uncertainty_mug_handle_w_0.02.npz", rgb=rgb, depth=depth, object_positions=object_poses_in_cam_frame.pos, object_quaternions=object_poses_in_cam_frame.quat)
