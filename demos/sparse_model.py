import jax.numpy as jnp
import jax
import trimesh
import rerun as rr
import b3d
import optax
from functools import partial
from tqdm import tqdm

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict.'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn
label_fn = map_nested_fn(lambda k, _: k)

box_mesh = trimesh.creation.box(jnp.ones(3))
box_vertices = jnp.array(box_mesh.vertices)

fx,fy = 100.0, 100.0
cx,cy = 50, 50
image_height, image_width = 100, 100
rr.init("demo")
rr.connect("127.0.0.1:8812")

def model(params, fx, fy, cx, cy):
    xyz_in_camera_frame = b3d.Pose(params["position"], params["quaternion"]).apply(params["xyz"])
    pixel_coords = b3d.xyz_to_pixel_coordinates(xyz_in_camera_frame, fx, fy, cx, cy)
    return pixel_coords

def pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[ jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)].set(jnp.arange(len(pixel_coords))+1 )
    return img



xyz = box_vertices
gt_pose = b3d.Pose.sample_gaussian_vmf_pose(jax.random.PRNGKey(10), b3d.Pose.from_translation(jnp.array([0.0, 0.0, 3.0])), 0.001, 1.0)
gt_pixel_coordinates = model({"xyz": xyz, "position": gt_pose.pos, "quaternion": gt_pose.quat}, fx,fy, cx,cy)


pose = b3d.Pose.sample_gaussian_vmf_pose(jax.random.PRNGKey(111), b3d.Pose.from_translation(jnp.array([0.0, 0.0, 3.0])), 0.001, 1.0)

def loss_function(params, gt_pixel_coordinates):
    pixel_coords = model(params, fx,fy, cx,cy)
    return jnp.mean((pixel_coords - gt_pixel_coordinates)**2)
loss_func_grad = jax.jit(jax.value_and_grad(loss_function, argnums=(0,)))

@partial(jax.jit, static_argnums=(0,))
def update_params(tx, params, gt_image, state):
    loss, (gradients,) = loss_func_grad(params, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


tx = optax.multi_transform(
    {
        'xyz': optax.sgd(0.0),
        'position': optax.adam(2e-2),
        'quaternion': optax.adam(2e-2),
    },
    label_fn
)
params = {
    "xyz": xyz,
    "position": pose.pos,
    "quaternion": pose.quat,
}
state = tx.init(params)

pixel_coords = model(params, fx,fy, cx,cy)
print(jnp.abs(pixel_coords - gt_pixel_coordinates))

rr.log("gt", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)), timeless=True)
rr.log("overlay", rr.DepthImage(pixel_coordinates_to_image(gt_pixel_coordinates, image_height, image_width)), timeless=True)

pbar = tqdm(range(300))
for t in pbar:
    params, state, loss = update_params(tx, params, gt_pixel_coordinates, state)
    pbar.set_description(f"Loss: {loss}")
    rr.set_time_sequence("frame", t)
    pixel_coords = model(params, fx,fy, cx,cy)
    rr.log("reconstruction", rr.DepthImage(pixel_coordinates_to_image(pixel_coords, image_height, image_width)))
    rr.log("overlay/reconstruction", rr.DepthImage(pixel_coordinates_to_image(pixel_coords, image_height, image_width)))

print(jnp.abs(pixel_coords - gt_pixel_coordinates))
print(jnp.abs(pixel_coords - gt_pixel_coordinates).max())

