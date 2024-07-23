import os
from functools import partial

import b3d
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rerun as rr
from matplotlib import colormaps
from tqdm import tqdm


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict."""

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


label_fn = map_nested_fn(lambda k, _: k)


# Load date
# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/static_room_pan_around.r3d.video_input.npz",
# )
# video_input = b3d.io.VideoInput.load(path)
# data = jnp.load(os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/cotracker_static_outputs.npz"
# ))

# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/bubly_manipulation.r3d.video_input.npz",
# )
# video_input = b3d.io.VideoInput.load(path)
# data = jnp.load(os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/cotracker_bubly_outputs.npz"
# ))

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/royce_static_to_dynamic.r3d.video_input.npz",
)
video_input = b3d.io.VideoInput.load(path)
data = jnp.load(path + "cotracker_output.npz")

first_frame_rgb = video_input.rgb[0] / 255.0

# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_rgb
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)

rr.init("demo")
rr.connect("127.0.0.1:8812")


def _pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[
        jnp.round(pixel_coords[:, 0]).astype(jnp.int32),
        jnp.round(pixel_coords[:, 1]).astype(jnp.int32),
    ].set(jnp.arange(len(pixel_coords)) + 1)
    return img


pixel_coordinates_to_image = jax.vmap(
    _pixel_coordinates_to_image, in_axes=(0, None, None)
)
pixel_coordinates_to_image_jit = jax.jit(pixel_coordinates_to_image)

sample_gaussian_vmf_multiple = jax.jit(
    jax.vmap(b3d.Pose.sample_gaussian_vmf_pose, in_axes=(0, None, None, None))
)


STRIDE = 5
pred_tracks = jnp.array(data["pred_tracks"])[0]
pred_visibility = jnp.array(data["pred_visibility"])[0]
gt_pixel_coordinates = pred_tracks[:, ..., jnp.array([1, 0])][::STRIDE]
gt_pixel_coordinate_colors = first_frame_rgb[
    jnp.round(gt_pixel_coordinates[0, :, 0]).astype(jnp.int32),
    jnp.round(gt_pixel_coordinates[0, :, 1]).astype(jnp.int32),
]
gt_visibility = pred_visibility[::STRIDE]
num_timesteps, num_keypoints = gt_pixel_coordinates.shape[:2]
rgbs = video_input.rgb[::3][::STRIDE]


def _model(params, cluster_assignments, fx, fy, cx, cy):
    xyz_relative_positions = params["xyz"]

    object_positions = params["object_positions"]
    object_quaternions = params["object_quaternions"]

    num_timesteps, _num_clusters, _ = object_positions.shape

    object_positions_expanded = jnp.concatenate(
        [jnp.zeros((num_timesteps, 1, 3)), object_positions], axis=1
    )
    object_quaternions_expanded = jnp.concatenate(
        [
            jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (num_timesteps, 1, 1)),
            object_quaternions,
        ],
        axis=1,
    )

    object_poses_over_time = b3d.Pose(
        object_positions_expanded, object_quaternions_expanded
    )
    xyz_in_world_frame_over_time = object_poses_over_time[:, cluster_assignments].apply(
        xyz_relative_positions
    )

    camera_poses_over_time = b3d.Pose(
        params["camera_positions"], params["camera_quaternions"]
    ).reshape((-1, 1))
    xyz_in_camera_frame_over_time = camera_poses_over_time.inv().apply(
        xyz_in_world_frame_over_time
    )
    pixel_coords = b3d.xyz_to_pixel_coordinates(
        xyz_in_camera_frame_over_time, fx, fy, cx, cy
    )
    return pixel_coords, xyz_in_world_frame_over_time


_model_jit = jax.jit(_model)


def model(params, cluster_assignments, fx, fy, cx, cy):
    return _model(params, cluster_assignments, fx, fy, cx, cy)[0]


model_jit = jax.jit(model)


def loss_function(t, params, cluster_assignments, gt_info):
    gt_pixel_coordinates, gt_visibility = gt_info
    pixel_coords = model(params, cluster_assignments, fx, fy, cx, cy)
    error = gt_visibility[..., None] * (pixel_coords - gt_pixel_coordinates) ** 2
    mask = jnp.arange(len(params["camera_positions"])) < t
    return jnp.mean(error * mask[..., None, None])


loss_func_grad = jax.jit(jax.value_and_grad(loss_function, argnums=(1,)))


@partial(jax.jit, static_argnums=(0, 1))
def update_params(tx, t, params, cluster_assignments, gt_image, state):
    loss, (gradients,) = loss_func_grad(t, params, cluster_assignments, gt_image)
    updates, state = tx.update(gradients, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss


def viz_params(params, start_t, end_t):
    _num_timesteps, num_clusters = params["object_positions"].shape[:2]
    num_keypoints = params["xyz"].shape[0]
    _, xyz_in_world_frame_over_time = _model(
        params, cluster_assignments, fx, fy, cx, cy
    )

    pixel_coords = model_jit(params, cluster_assignments, fx, fy, cx, cy)
    error_per_timestep = (
        jnp.abs(pixel_coords - gt_pixel_coordinates) * gt_visibility[..., None]
    ).sum(-1)

    for t in range(start_t, end_t):
        for c in range(num_clusters):
            rr.set_time_sequence("frame", t)
            b3d.rr_log_pose(
                f"/object_{c}",
                b3d.Pose(
                    params["object_positions"][t, c], params["object_quaternions"][t, c]
                ),
            )
        rr.set_time_sequence("frame", t)

        camera_pose = b3d.Pose(
            params["camera_positions"][t], params["camera_quaternions"][t]
        )
        rr.log(
            "/camera",
            rr.Transform3D(
                translation=camera_pose.position,
                rotation=rr.Quaternion(xyzw=camera_pose.xyzw),
            ),
        )
        rr.log(
            "/camera",
            rr.Pinhole(
                resolution=[0.1, 0.1],
                focal_length=0.1,
            ),
        )
        rr.log(
            "xyz",
            rr.Points3D(
                xyz_in_world_frame_over_time[t], colors=gt_pixel_coordinate_colors
            ),
        )
        redness = (
            jnp.tile(jnp.array([1.0, 0.0, 0.0]), (num_keypoints, 1))
            * jnp.clip(error_per_timestep[t] / 20.0, 0.0, 1.0)[..., None]
        )
        rr.log(
            "xyz/error", rr.Points3D(xyz_in_world_frame_over_time[t], colors=redness)
        )
        # rr.log("rgb", rr.Image(rgbs[t] / 255.0))


colors = colormaps["rainbow"](jnp.linspace(0, 1, len(gt_pixel_coordinates[0])))
# for t in range(len(pred_tracks)):
#     rr.set_time_sequence("frame", t)
#     rr.log("rgb", rr.Image(video_input.rgb[::3][t] / 255.0))

#     rr.log("rgb/points", rr.Points2D(jnp.transpose(pred_tracks[t,...,jnp.array([0,1])],(1,0))))
#                                 #  colors=colors * pred_visibility[t][:,None] + (1.0 - pred_visibility[t])[:,None]))


#     rr.log("rgb/points", rr.Points2D(jnp.transpose(pred_tracks[t,...,jnp.array([0,1])],(1,0)),
#                                  colors=colors * pred_visibility[t][:,None] + (1.0 - pred_visibility[t])[:,None]))


# pil_images = [
#     b3d.get_rgb_pil_image(rgb / 255.0)
#     for rgb in video_input.rgb[:200][::T]
# ]
# b3d.make_video_from_pil_images(pil_images, "video.mp4")

num_clusters = 1
camera_poses = sample_gaussian_vmf_multiple(
    jax.random.split(jax.random.PRNGKey(1111110), num_timesteps),
    b3d.Pose.from_translation(jnp.array([0.0, 0.0, 6.0])),
    0.01,
    0.01,
).inv()
object_poses_over_time = sample_gaussian_vmf_multiple(
    jax.random.split(jax.random.PRNGKey(1111110), num_timesteps * num_clusters),
    b3d.Pose.from_translation(jnp.array([0.0, 0.0, 0.0])),
    0.01,
    0.01,
).reshape((num_timesteps, num_clusters))
cluster_assignments = jnp.zeros((num_keypoints,), dtype=jnp.int32)
params = {
    "xyz": jax.random.uniform(jax.random.PRNGKey(1000), (num_keypoints, 3)) * 0.1,
    "camera_positions": camera_poses.pos,
    "camera_quaternions": camera_poses.quat,
    "object_positions": object_poses_over_time.pos,
    "object_quaternions": object_poses_over_time.quat,
}

tx = optax.multi_transform(
    {
        "xyz": optax.adam(1e-3),
        "camera_positions": optax.adam(1e-2),
        "camera_quaternions": optax.adam(1e-2),
        "object_positions": optax.adam(1e-2),
        "object_quaternions": optax.adam(1e-2),
    },
    label_fn,
)

rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)

state = tx.init(params)

END_T = len(gt_pixel_coordinates)
pbar = tqdm(range(3000))
for t in pbar:
    params, state, loss = update_params(
        tx,
        END_T,
        params,
        cluster_assignments,
        (gt_pixel_coordinates, gt_visibility),
        state,
    )
    pbar.set_description(f"Loss: {loss}")
    params_over_time.append(params)
viz_params(params, 0, END_T)


END_T = len(gt_pixel_coordinates)
pbar = tqdm(range(3000))
for t in pbar:
    params, state, loss = update_params(
        tx,
        5,
        params,
        cluster_assignments,
        (gt_pixel_coordinates[:5], gt_visibility[:5]),
        state,
    )
    pbar.set_description(f"Loss: {loss}")
    params_over_time.append(params)
viz_params(params, 0, END_T)


INITIAL_T = END
pbar = tqdm(range(3000))
for t in pbar:
    params, state, loss = update_params(
        tx,
        INITIAL_T,
        params,
        cluster_assignments,
        (gt_pixel_coordinates, gt_visibility),
        state,
    )
    pbar.set_description(f"Loss: {loss}")
    params_over_time.append(params)
viz_params(params, 0, INITIAL_T)

FIRST_T = 33

for running_t in range(INITIAL_T, FIRST_T + 1):
    pbar = tqdm(range(300))
    for t in pbar:
        params, state, loss = update_params(
            tx,
            running_t,
            params,
            cluster_assignments,
            (gt_pixel_coordinates, gt_visibility),
            state,
        )
        pbar.set_description(f"Loss: {loss}")
        params_over_time.append(params)
    viz_params(params, running_t, running_t + 1)


END_T = len(gt_pixel_coordinates)

params_over_time = [params]
pbar = tqdm(range(2000))
for t in pbar:
    params, state, loss = update_params(
        tx,
        SECOND_T,
        params,
        cluster_assignments,
        (gt_pixel_coordinates, gt_visibility),
        state,
    )
    pbar.set_description(f"Loss: {loss}")
    params_over_time.append(params)
viz_params(params, FIRST_T, SECOND_T)

pixel_coords = model_jit(params, cluster_assignments, fx, fy, cx, cy)
reconstruction = pixel_coordinates_to_image(pixel_coords, image_height, image_width)[0]
error = jnp.abs(pixel_coords - gt_pixel_coordinates).sum(-1) * gt_visibility
average_error = error.sum(0) / gt_visibility.sum(0)
top_indices = jnp.argsort(-average_error)[:150]
rr.log(
    "xyz/grouping",
    rr.Points3D(
        params["xyz"][top_indices], colors=gt_pixel_coordinate_colors[top_indices]
    ),
)

print(loss_function(SECOND_T, params, cluster_assignments, gt_info))

cluster_assignments = cluster_assignments.at[top_indices].set(
    cluster_assignments.max() + 1
)
mean_xyz = params["xyz"][top_indices].mean(0)
params["xyz"] = params["xyz"].at[top_indices].set(params["xyz"][top_indices] - mean_xyz)
params["object_positions"] = params["object_positions"].at[:, 0, :].set(mean_xyz)
params["object_quaternions"] = (
    params["object_quaternions"].at[:, 0, :].set(jnp.array([0.0, 0.0, 0.0, 1.0]))
)


pbar = tqdm(range(2000))
for t in pbar:
    params, state, loss = update_params(
        tx,
        END_T,
        params,
        cluster_assignments,
        (gt_pixel_coordinates, gt_visibility),
        state,
    )
    pbar.set_description(f"Loss: {loss}")
    params_over_time.append(params)
viz_params(params, 0, END_T)


viz_params(params)
