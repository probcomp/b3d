import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm   import tqdm
import trimesh

PORT = 8812
rr.init("real")
rr.connect(addr=f'127.0.0.1:{PORT}')

video_input = b3d.VideoInput.load(os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/input_data/mug_handle_occluded.video_input.npz"
# "assets/shared_data_bucket/input_data/mug_handle_visible.video_input.npz"
))

scaling_factor = 4
image_width, image_height, fx,fy, cx,cy,near,far = jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
image_width, image_height = int(image_width), int(image_height)
fx,fy, cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

_rgb = video_input.rgb[0].astype(jnp.float32) / 255.0
_depth = video_input.xyz[0].astype(jnp.float32)[...,2]
rgb = jnp.clip(jax.image.resize(
    _rgb, (image_height, image_width, 3), "nearest"
), 0.0, 1.0)
depth = jax.image.resize(
    _depth, (image_height, image_width), "nearest"
)

point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1,3)
rr.log("point_cloud", rr.Points3D(point_cloud.reshape(-1,3), colors=rgb.reshape(-1,3)))
table_pose, table_dims = b3d.Pose.fit_table_plane(point_cloud, 0.01, 0.01, 100, 1000)
b3d.rr_log_pose("table", table_pose)

mesh_path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/025_mug/textured.obj")
mesh = trimesh.load(mesh_path)
vertices = jnp.array(mesh.vertices)
vertices = vertices - jnp.mean(vertices, axis=0)
faces = jnp.array(mesh.faces)
vertex_colors = (jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0 ) 

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)
model = b3d.model_multiobject_gl_factory(renderer)
importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(0)


rgb_object_samples = vertex_colors[jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))]
distances = jnp.abs(rgb[...,None] - rgb_object_samples.T).sum([-1,-2])
rr.log("image/distances", rr.DepthImage(distances))

object_center_hypothesis = point_cloud[distances.argmin()]



color_error, depth_error = (50.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (2000.0, 2000.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,

    inlier_score,
    outlier_prob,

    color_multiplier,
    depth_multiplier,
)

key = jax.random.PRNGKey(0)


from functools import partial
@partial(jax.jit, static_argnames=['addresss', 'number'])
def gvmf_pose_proposal(trace, key, variance, concentration, address, number):
    addr = address.const
    test_poses = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
        jax.random.split(key, number),
        trace[addr],
        variance, concentration
    ), trace[addr][None,...]])
    scores = b3d.enumerate_choices_get_scores(
        trace, jax.random.PRNGKey(0), genjax.Pytree.const([addr]), test_poses
    )
    sample = jax.random.categorical(key, scores)
    trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const([addr]),
            test_poses[sample])
    key = jax.random.split(key, 2)[-1]
    return trace, key


trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.sample_gaussian_vmf_pose(
                key, Pose.from_translation(object_center_hypothesis), 0.001, 0.01),
            "object_0": 0,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    (jnp.arange(1), model_args, object_library)
)
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))

for t in tqdm(range(40)):
    trace, key = gvmf_pose_proposal(trace, key,
        0.01, 10.0, genjax.Pytree.const("object_pose_0"), 1000)


print(trace.get_score())
b3d.rerun_visualize_trace_t(trace, t)


