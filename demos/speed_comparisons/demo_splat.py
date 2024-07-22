import jax.numpy as jnp
import jax
import numpy as np
import os
import b3d
from b3d import Pose
import rerun as rr
from tqdm import tqdm
from diff_gaussian_rasterization import rasterize_with_depth

rr.init("demo")
rr.connect("127.0.0.1:8812")

# Load date
path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.io.VideoInput.load(path)

# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_depth
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

# Get RGBS and Depth
rgbs = video_input.rgb[::4] / 255.0
xyzs = video_input.xyz[::4]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)


point_cloud = jax.image.resize(
    xyzs[0], (xyzs[0].shape[0] // 2, xyzs[0].shape[1] // 2, 3), "linear"
).reshape(-1, 3)
colors = jax.image.resize(
    rgbs_resized[0], (xyzs[0].shape[0] // 2, xyzs[0].shape[1] // 2, 3), "linear"
).reshape(-1, 3)
wxyz = Pose.identity()[None, ...].wxyz


def render_rgb(point_cloud, colors, scale, camera_pose):
    rendered_image, rendered_depth = rasterize_with_depth(
        camera_pose.inv().apply(point_cloud),
        colors,
        jnp.ones((len(point_cloud), 1)),
        jnp.ones((len(point_cloud), 3)) * scale,
        jnp.tile(wxyz, (len(point_cloud), 1)),
        image_width,
        image_height,
        fx,
        fy,
        cx,
        cy,
        near,
        far,
    )
    return jnp.permute_dims(rendered_image, (1, 2, 0)), rendered_depth


render_rgb_jit = jax.jit(render_rgb)


def loss_fun(point_cloud, colors, scale, camera_pose, observed_image, observed_depth):
    rendered_image, rendered_depth = render_rgb(point_cloud, colors, scale, camera_pose)
    return jnp.mean(jnp.abs((rendered_image - observed_image)))
    # #  + jnp.mean(
    #     jnp.abs((rendered_depth - observed_depth))
    # )


loss_grad = jax.jit(
    jax.value_and_grad(
        loss_fun,
        argnums=(
            0,
            1,
            2,
        ),
    )
)
loss_grad_camera_pose = jax.jit(jax.value_and_grad(loss_fun, argnums=(3,)))

scale = jnp.float32(0.005)
camera_pose = Pose.identity()
t = 0
gt_image = rgbs_resized[t]
gt_depth = xyzs[t, ..., 2]
loss_grad(point_cloud, colors, scale, Pose.identity(), gt_image, gt_depth)

rr.log("/image", rr.Image(gt_image), timeless=True)

pbar = tqdm(range(100))
for t in pbar:
    loss, (grad_point_cloud, grad_colors, grad_scale) = loss_grad(
        point_cloud, colors, scale, camera_pose, gt_image, gt_depth
    )
    scale = scale - grad_scale * 0.001
    # point_cloud = point_cloud - grad_point_cloud * 0.001
    grad_colors = grad_colors - grad_colors * 0.001
    pbar.set_description(f"Loss: {loss}")
    rr.set_time_sequence("/frame", t)
    rr.log(
        "/reconstruction",
        rr.Image(render_rgb_jit(point_cloud, colors, scale, camera_pose)[0]),
    )
    rr.log(
        "/image/reconstruction",
        rr.Image(render_rgb_jit(point_cloud, colors, scale, camera_pose)[0]),
    )


t = 200
gt_image = rgbs_resized[t]
gt_depth = xyzs[t, ..., 2]
rr.log("/image", rr.Image(gt_image), timeless=True)

pbar = tqdm(range(100))
for t in pbar:
    loss, (grad_camera_pose,) = loss_grad_camera_pose(
        point_cloud, colors, scale, camera_pose, gt_image, gt_depth
    )
    camera_pose = camera_pose - grad_camera_pose * 0.0001
    pbar.set_description(f"Loss: {loss}")
    rr.set_time_sequence("/frame", t)
    rr.log(
        "/reconstruction",
        rr.Image(render_rgb_jit(point_cloud, colors, scale, camera_pose)[0]),
    )
    rr.log(
        "/image/reconstruction",
        rr.Image(render_rgb_jit(point_cloud, colors, scale, camera_pose)[0]),
    )


camera_pose = Pose.identity()
for t in tqdm(range(150)):
    rr.set_time_sequence("frame", t)
    gt_image = rgbs_resized[t]
    gt_depth = xyzs[t, ..., 2]
    rr.log("/image/actual", rr.Image(gt_image))
    for _ in range(100):
        loss, grad = loss_grad(camera_pose, gt_image, gt_depth)
        camera_pose = camera_pose - grad * 0.005
    rr.log("/image", rr.Image(render_rgb_jit(camera_pose)[0]))


rasterize(*x, 200, 200, 200.0, 200.0, 100.0, 100.0, 0.01, 10.0)


# Take point cloud at frame 0


num_layers = 2048
renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)

# Arguments of the generative model.
# These control the inlier / outlier decision boundary for color error and depth error.
color_error, depth_error = (jnp.float32(30.0), jnp.float32(0.02))
# TODO: explain
inlier_score, outlier_prob = (jnp.float32(5.0), jnp.float32(0.001))
# TODO: explain
color_multiplier, depth_multiplier = (jnp.float32(3000.0), jnp.float32(3000.0))
