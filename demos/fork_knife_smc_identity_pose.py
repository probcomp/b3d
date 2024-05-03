import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm import tqdm
import trimesh
import genjax
import matplotlib.pyplot as plt 

### Choose experiment

INPUT = "fork-visible"  # TODO make one dataset with both objects?
INPUT = "knife-visible"
INPUT = "fork-occluded"
# INPUT = "knife-occluded"

PORT = 8812
rr.init(f"fork-knife2_{INPUT}")
rr.connect(addr=f"127.0.0.1:{PORT}")


############
# Load data
############

print(f"Running with input {INPUT}")
if "fork" in INPUT:
    video_input = b3d.VideoInput.load(b3d.get_root_path() / "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_fork.npz")
    if "visible" in INPUT: 
        T = 1
    elif "occluded" in INPUT: 
        T = 8    
elif "knife" in INPUT:
    video_input = b3d.VideoInput.load(b3d.get_root_path() / "assets/shared_data_bucket/datasets/identity_uncertainty_fork_knife_knife.npz")
    if "visible" in INPUT: 
        T = 1
    elif "occluded" in INPUT: 
        T = 0   
else:
    raise ValueError(f"Unknown input {INPUT}")


### Build meshes / object library 

fork_path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/030_fork/textured.obj",
)
knife_path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/032_knife/textured.obj",
)

FORK_ID = 0; KNIFE_ID = 1
object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(trimesh.load(fork_path))
object_library.add_trimesh(trimesh.load(knife_path))

occluder = trimesh.creation.box(extents=np.array([0.15, 0.1, 0.02]))
occluder_colors = jnp.tile(jnp.array([0.8, 0.8, 0.8])[None,...], (occluder.vertices.shape[0], 1))

_camera_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.0, 0.2, 0.8]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0])
)
occluder_pose_in_camera_frame = _camera_pose.inv() @ b3d.Pose.from_pos(jnp.array([0.0, 0.05, 0.3]))

object_library.add_object(occluder.vertices, occluder.faces, attributes=occluder_colors)

print(f"{object_library.get_num_objects()} objects in library")





T = 10
T = 1

scaling_factor = 2
image_width, image_height, fx, fy, cx, cy, near, far = (
    jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
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

image_width, image_height, fx, fy, cx, cy, near, far = (
    jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
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

_rgb = video_input.rgb[T].astype(jnp.float32) / 255.0
_depth = video_input.xyz[T].astype(jnp.float32)[..., 2]
rgb = jnp.clip(
    jax.image.resize(_rgb, (image_height, image_width, 3), "nearest"), 0.0, 1.0
)
depth = jax.image.resize(_depth, (image_height, image_width), "nearest")
rr.log("rgb", rr.Image(rgb))


###############
# Setup model + renderer
###############

### Inference params



renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer, b3d.rgbd_sensor_model)
importance_jit = jax.jit(model.importance)
key = jax.random.PRNGKey(110)

### initialize pose hypothesis just based on depth info 

# masked_depth = depth * (depth != scene_depth)
# point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy)
# valid_point_cloud = point_cloud[depth != scene_depth].reshape(-1, 3)  # cloud for object only
# object_center_hypothesis = valid_point_cloud.mean(axis=0)
# print(object_center_hypothesis)


point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1, 3)

vertex_colors = object_library.attributes
rgb_object_samples = vertex_colors[
    jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))
]
distances = jnp.abs(rgb[..., None] - rgb_object_samples.T).sum([-1, -2]) + 100.0 * (depth == 0.0)
rr.log("image/distances", rr.DepthImage(distances))
print("best index", jnp.unravel_index(distances.argmin(), distances.shape))
# rr.log("img", rr.Image(rgb))

object_center_hypothesis = point_cloud[distances.argmin()]

rr.log("point_cloud", rr.Points3D(point_cloud))

### Initialize trace


color_error, depth_error = (60.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.00001)
color_multiplier, depth_multiplier = (5000.0, 500.0)
model_args = b3d.ModelArgs(
    color_error,
    depth_error,
    inlier_score,
    outlier_prob,
    color_multiplier,
    depth_multiplier,
)

key = jax.random.split(key, 2)[-1]
variance = 0.01
concentration = 0.01
trace, _ = importance_jit(
    jax.random.PRNGKey(10),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.sample_gaussian_vmf_pose(
                key, Pose.from_translation(object_center_hypothesis), 
                0.00001, concentration,   
                # TODO
                # if rotation is not tightly constrained with a high concentration 
                # for this initial vmf sampling (which can generate any angle)
                # then it is hard to correct a large deviation via gridding, 
                # which is restricted to contact angles
            ),
            "object_pose_1": occluder_pose_in_camera_frame,
            "object_1": 2,
            "observed_rgb_depth": (rgb, depth),
        }
    ),
    (jnp.arange(2), model_args, object_library),
)

### initialize trace corresponding to each object; will do c2f on these
trace_fork = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_0"]), FORK_ID) 
trace_knife = b3d.update_choices_jit(trace, key, genjax.Pytree.const(["object_0"]), KNIFE_ID)
b3d.rerun_visualize_trace_t(trace_fork, 0)
b3d.rerun_visualize_trace_t(trace_knife, 1)


################
# C2F
################
## incrementally update pose and weights for each object hypothesis

init_params = jnp.array([0.01, 10.0])
num_samples = 5000

def gvmf_c2f(trace, key, params):
    skips = 0
    for t in tqdm(range(30)):    
        (
            trace_new_pose,
            key,
        ) = b3d.gvmf_and_sample(
            trace, key, params[0], params[1], 
            genjax.Pytree.const("object_pose_0"), num_samples
        )
        
        if trace_new_pose.get_score() > trace.get_score():
            trace = trace_new_pose
            # b3d.rerun_visualize_trace_t(trace, 0)
            print(f"new: {trace['object_pose_0'].pos}")
        else:
            params = jnp.array([params[0] * 0.5, params[1] * 2.0])
            skips += 1
            print(f"shrinking")
            if skips > 5:
                print(f"skip {t}")
                break
    return trace, key
        
trace_fork, key = gvmf_c2f(trace_fork, key, init_params)
trace_knife, key = gvmf_c2f(trace_knife, key, init_params)
b3d.rerun_visualize_trace_t(trace_fork, 0)
b3d.rerun_visualize_trace_t(trace_knife, 1)


### Setup grid for sampling
delta_cps = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.05, 0.05, 41),
        jnp.linspace(-0.05, 0.05, 41),
        jnp.linspace(-jnp.pi, jnp.pi, 71),
    ),
    axis=-1,
).reshape(-1, 3)

contact_parameters_to_pose_camspace = lambda cp: b3d.Pose(
    jnp.array([cp[0], 0.0, cp[1]]),  # fixed height (y) at 0 for table 
    b3d.Rot.from_rotvec(jnp.array([0.0, cp[2], 0.0])).as_quat(),
)
cp_delta_poses = jax.vmap(contact_parameters_to_pose_camspace)(delta_cps)


def grid_c2f(trace, key):
    for _ in range(2):
        key = jax.random.split(key, 2)[-1]

        test_poses = trace["object_pose_0"] @ cp_delta_poses
        test_poses_batches = test_poses.split(20)

        scores = jnp.concatenate(
            [
                b3d.enumerate_choices_get_scores_jit(
                    trace, key, genjax.Pytree.const(["object_pose_0"]), poses
                )
                for poses in test_poses_batches
            ]
        )
        print("Score Max", scores.max())
        samples = jax.random.categorical(key, scores, shape=(50,))

        samples_deg_range = jnp.rad2deg(
            (
                jnp.max(delta_cps[samples], axis=0)
                - jnp.min(delta_cps[samples], axis=0)
            )[2]
        )

        trace = b3d.update_choices_jit(
            trace,
            key,
            genjax.Pytree.const(["object_pose_0"]),
            test_poses[samples[0]],
        )
        print("Sampled Angle Range:", samples_deg_range)
        
        return trace, key, test_poses, samples

trace_fork, key, test_poses, samples = grid_c2f(trace_fork, key)
trace_knife, key, test_poses, samples = grid_c2f(trace_knife, key)
b3d.rerun_visualize_trace_t(trace_fork, 0)
b3d.rerun_visualize_trace_t(trace_knife, 1)
print("Normalized scores: ", b3d.normalize_log_scores(jnp.array([trace_fork.get_score(), trace_knife.get_score()])))

# for i in range(2):
#     rr.log(
#         f"/3d/mesh/{i}",
#         rr.Mesh3D(
#             vertex_positions=(object_library.vertices),
#             indices=object_library.faces[object_library.ranges[i,0]: object_library.ranges[i,:].sum()],
#             vertex_colors=object_library.attributes
#         )
#     )



# ############
# # visualize samples
# ############

# for t in range(len(fork_samples)):
    
#     # fork posterior samples
#     _fork_viz = b3d.update_choices_jit(
#         trace_fork,
#         key,
#         genjax.Pytree.const(["object_pose_0"]),
#         fork_test_poses[fork_samples[t]],
#     )
#     b3d.rerun_visualize_trace_t(_fork_viz, 2*t)
    
#     # knife posterior samples 
#     _knife_viz = b3d.update_choices_jit(
#         trace_knife,
#         key,
#         genjax.Pytree.const(["object_pose_0"]),
#         knife_test_poses[knife_samples[t]],
#     )
#     b3d.rerun_visualize_trace_t(_knife_viz, 2*t+1)
    
    
# from IPython import embed; embed()