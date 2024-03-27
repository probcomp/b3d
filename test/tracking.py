import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm

PORT = 8813
rr.init("bunny")
rr.connect(addr=f'127.0.0.1:{PORT}')

path = os.path.join(b3d.get_assets_path(),
#  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
 "shared_data_bucket/input_data/ramen_case.r3d.video_input.npz")
video_input = b3d.VideoInput.load(path)

image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

rgbs = video_input.rgb / 255.0
# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
), 0.0, 1.0)



import torch
from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=231,
                        trimap_dilation=30,
                        trimap_erosion_iters=5,
                        fp16=False)

T = 0
output_images = interface([b3d.get_rgb_pil_image(rgbs_resized[T])])
mask  = jnp.array([jnp.array(output_image)[..., -1] > 0.5 for output_image in output_images])[0]

rr.log("/img", rr.Image(rgbs_resized[T]))
rr.log("/img/mask", rr.Image(jnp.tile((mask * 1.0)[...,None],(1,1,3))))

xyz = video_input.xyz[0]
point_cloud = xyz[mask]
colors = rgbs_resized[0][mask]

subset = (point_cloud < point_cloud.mean(0)).all(1)
point_cloud = point_cloud[subset]
colors = colors[subset]



vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud, colors, 0.003 * 2 * jnp.ones(len(colors))
)
object_pose = Pose.from_translation(vertices.mean(0))
vertices = object_pose.inverse().apply(vertices)

rr.log(
    "/3d/mesh",
    rr.Mesh3D(
        vertex_positions=vertices,
        indices=faces,
        vertex_colors=vertex_colors
    ),
    timeless=True
)


renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
from b3d.model import model_gl_factory

# image = renderer.render_attribute(
#     object_pose.as_matrix()[None,...], vertices, faces, jnp.array([[0, len(faces)]]), vertex_colors
# )
# rr.log("/img/rerender", rr.Image(image))



model = model_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
update_jit = jax.jit(model.update)


enumerator = b3d.make_enumerator(["object_pose"])
enumerator_observations = b3d.make_enumerator(["observed_rgb", "observed_depth"])


translation_deltas = jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.01, 0.01, 5),
        jnp.linspace(-0.01, 0.01, 5),
        jnp.linspace(-0.01, 0.01, 5),
    ),
    axis=-1,
).reshape(-1, 3))

rotation_deltas = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 100),
    Pose.identity(),
    0.00001, 1000.0
)


@jax.jit
def enumerative_proposal(trace, key):
    key = jax.random.split(key)[0]

    test_poses = trace["object_pose"] @ translation_deltas
    potential_scores = enumerator.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )

    test_poses = trace["object_pose"] @ rotation_deltas
    potential_scores = enumerator.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), test_poses
    )
    trace = enumerator.update_choices(
        trace, key, test_poses[potential_scores.argmax()]
    )
    return trace, key



color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (4.0, 0.01)
color_multiplier, depth_multiplier = (1000.0, 1000.0)
arguments = (
        vertices, faces, vertex_colors,
        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier
    )


START_T = 0
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose": object_pose,
            "observed_rgb": rgbs_resized[START_T],
            "observed_depth": video_input.xyz[START_T,...,2],
        }
    ),
    arguments
)


b3d.rerun_visualize_trace_t(trace, 0)

END_T = len(video_input.xyz)
key = jax.random.PRNGKey(0)
chain2 = []
for T_observed_image in tqdm(range(START_T,END_T, 1)):
    trace = enumerator_observations.update_choices(trace, key,
        rgbs_resized[T_observed_image],
        video_input.xyz[T_observed_image,...,2]
    )
    for _ in range(1):
        trace,key = enumerative_proposal(trace, key)
    chain2.append(trace["object_pose"])

for T_observed_image in tqdm(range(len(chain2))):
    trace = enumerator_observations.update_choices(trace, key,
        rgbs_resized[T_observed_image],
        video_input.xyz[T_observed_image,...,2]
    )
    trace = enumerator.update_choices(trace, key, chain2[T_observed_image])
    b3d.rerun_visualize_trace_t(trace, T_observed_image)
    rr.log("/point_cloud",
        rr.Points3D(
            video_input.xyz[T_observed_image].reshape(-1, 3),
        )
    )
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    rr.log("/point_cloud",
        rr.Points3D(
            video_input.xyz[T_observed_image].reshape(-1, 3),
        )
    )
    rr.log("/rendered_point_cloud",
        rr.Points3D(
            b3d.xyz_from_depth(rendered_depth, fx, fy, cx, cy).reshape(-1, 3),
        )
    )

from IPython import embed; embed()
