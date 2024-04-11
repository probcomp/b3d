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

import pickle
path = os.path.join(b3d.get_root_path(),
"assets/shared_data_bucket/input_data/mug2.pkl")
with open(path, "rb") as f:
    data = pickle.load(f)

scaling_factor = 4
_rgb = jnp.array(data[0]["camera_image"]["rgbPixels"]/255.0)
_depth = jnp.array(data[0]["camera_image"]["depthPixels"])
rgb = jnp.clip(jax.image.resize(
    _rgb, (_rgb.shape[0] // scaling_factor , _rgb.shape[1] // scaling_factor, 3), "nearest"
), 0.0, 1.0)
depth = jax.image.resize(
    _depth, (_depth.shape[0] // scaling_factor , _depth.shape[1] // scaling_factor), "nearest"
)

image_height, image_width = rgb.shape[:2]

K = data[0]["camera_image"]["camera_matrix"][0] / scaling_factor
fx,fy,cx,cy = K[0,0],K[1,1],K[0,2], K[1,2]

point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1,3)
rr.log("rgb", rr.Image(rgb))
rr.log("depth", rr.DepthImage(depth))
rr.log("points", rr.Points3D(point_cloud.reshape(-1,3), colors=rgb.reshape(-1,3)))

table_pose, table_dims = b3d.Pose.fit_table_plane(point_cloud, 0.01, 0.01, 100, 1000)


def rr_log_pose(channel, pose):
    origins = jnp.tile(pose.pos[None,...], (3,1))
    vectors = jnp.eye(3)
    colors = jnp.eye(3)
    rr.log(channel, rr.Arrows3D(origins=origins, vectors=pose.as_matrix()[:3,:3].T, colors=colors))

rr_log_pose("table", table_pose)

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


color_error, depth_error = (40.0, 0.03)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (700.0, 500.0)
arguments = (
        jnp.arange(1),
        color_error,
        depth_error,

        inlier_score,
        outlier_prob,

        color_multiplier,
        depth_multiplier,
        object_library
    )


key = jax.random.PRNGKey(0)
trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.sample_gaussian_vmf_pose(
                jax.random.PRNGKey(1200), table_pose, 0.01, 0.01),
            "object_0": 0,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    arguments
)
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))

rr_log_pose("object_pose", trace["object_pose_0"])



key = jax.random.split(key, 2)[-1]
rotation_deltas = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 11*11*11),
    Pose.identity(),
    0.01, 10.0
), Pose.identity()[None,...]])
test_poses = trace["object_pose_0"] @ rotation_deltas
scores = b3d.enumerate_choices_get_scores(
    trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]), test_poses
)
print(trace.get_score(), scores.max())
trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]),
        test_poses[scores.argmax()])
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))



delta_cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.02, 0.02, 21),
        jnp.linspace(-0.02, 0.02, 21),
        jnp.linspace(-jnp.pi, jnp.pi, 31),
    ),
    axis=-1,
).reshape(-1, 3)
cp_delta_poses = jax.vmap(contact_param_to_pose)(delta_cps) 


test_poses = trace["object_pose_0"] @ cp_delta_poses
test_poses_batches = test_poses.split(10)
scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(trace, key, genjax.Pytree.const(["object_pose_0"]), poses) for poses in test_poses_batches])
print(trace.get_score(), scores.max())
trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]),
        test_poses[scores.argmax()])
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))





translation_deltas = Pose.concatenate_poses([jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.1, 0.1, 7),
        jnp.linspace(-0.1, 0.1, 7),
        jnp.linspace(-0.1, 0.1, 7),
    ),
    axis=-1,
).reshape(-1, 3)), Pose.identity()[None,...]])
test_poses = trace["object_pose_0"] @ translation_deltas
scores = b3d.enumerate_choices_get_scores(
    trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]), test_poses
)
print(trace.get_score(), scores.max())
trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]),
        test_poses[scores.argmax()])
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))

rotation_deltas = Pose.concatenate_poses([jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), 11*11*11),
    Pose.identity(),
    0.01, 10.0
), Pose.identity()[None,...]])
test_poses = trace["object_pose_0"] @ rotation_deltas
scores = b3d.enumerate_choices_get_scores(
    trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]), test_poses
)
print(trace.get_score(), scores.max())
trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), genjax.Pytree.const(["object_pose_0"]),
        test_poses[scores.argmax()])
b3d.rerun_visualize_trace_t(trace, 0)
rr.log("prediction", rr.Points3D(trace["object_pose_0"].apply(vertices)))


contact_param_to_pose = lambda cp: Pose(
    jnp.array([cp[0], cp[1], 0.0]),
    b3d.Rot.from_rotvec(jnp.array([0.0, 0.0, cp[2]])).as_quat()
)


