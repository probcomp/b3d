import rerun as rr
import genjax
import os
import numpy as np
import jax.numpy as jnp
import jax
from b3d import Pose
import b3d
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt 
# from carvekit.api.high import HiInterface  # we dont need no neural netz here

###################################
# Setup
###################################

PORT = 8813
rr.init("SHOUT")
rr.connect(addr=f'127.0.0.1:{PORT}')

## get data (43*5 to end (228*5) of original full video)
subsampling_frame = 5
path = os.path.join(b3d.get_assets(), "shared_data_bucket/input_data/demo_reel_place_and_pickup.r3d.video_input_shout_to_end.npz")
video_input = b3d.VideoInput.load(path) 

image_width, image_height, fx,fy, cx,cy,near,far = np.array(video_input.camera_intrinsics_depth)
image_width, image_height = int(image_width), int(image_height)
fx,fy,cx,cy,near,far = float(fx),float(fy), float(cx),float(cy),float(near),float(far)

rgbs_all = video_input.rgb / 255.0
xyzs = video_input.xyz[::subsampling_frame, ...]
rgbs = rgbs_all[::subsampling_frame, ...]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(jax.vmap(jax.image.resize, in_axes=(0, None, None))(
    rgbs, (xyzs.shape[1], xyzs.shape[2], 3), "linear"
), 0.0, 1.0)

del video_input 
del rgbs_all

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
model = b3d.model_multiobject_gl_factory(renderer)

###################################
# Relevant time markers + params
###################################
BG_MESH_T = START_CAM_T = 0  # when to acquire background mesh
FULL_SHOUT_T = START_JOINT_T = 32  # first frame where full shout object is shown, no hand
END_T = rgbs_resized.shape[0]

object_library = b3d.model.MeshLibrary.make_empty_library()
poses = []

color_error, depth_error = (30.0, 0.02)
inlier_score, outlier_prob = (5.0, 0.01)
color_multiplier, depth_multiplier = (3000.0, 3000.0)
num_objects = 1  # one object (background) initially

###################################
# Inference setup
###################################
key = jax.random.PRNGKey(0)
NUM_ENUM = 5 # ideally >10
translation_deltas = jax.vmap(lambda p: Pose.from_translation(p))(jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-0.01, 0.01, NUM_ENUM),
        jnp.linspace(-0.01, 0.01, NUM_ENUM),
        jnp.linspace(-0.01, 0.01, NUM_ENUM),
    ),
    axis=-1,
).reshape(-1, 3))

rotation_deltas = jax.vmap(Pose.sample_gaussian_vmf_pose, in_axes=(0,None, None, None))(
    jax.random.split(jax.random.PRNGKey(0), NUM_ENUM*NUM_ENUM*NUM_ENUM),
    Pose.identity(),
    0.00001, 1000.0
)

all_deltas =  Pose.stack_poses([translation_deltas, rotation_deltas])

from functools import partial
@partial(jax.jit, static_argnames=['addressses'])
def enumerative_proposal(trace, addressses, key, all_deltas):
    addr = addressses.const[0]
    current_pose = trace[addr]
    for i in range(len(all_deltas)):
        test_poses = current_pose @ all_deltas[i]
        potential_scores = b3d.enumerate_choices_get_scores(
            trace, jax.random.PRNGKey(0), addressses, test_poses
        )
        current_pose = test_poses[potential_scores.argmax()]
    trace = b3d.update_choices(
        trace, key, addressses, current_pose
    )
    return trace, key

arguments = (jnp.empty((num_objects,)),color_error,depth_error,inlier_score,outlier_prob,color_multiplier,depth_multiplier, object_library)


###################################
# Background mesh acquisition
###################################
point_cloud_for_mesh = xyzs[BG_MESH_T].reshape(-1, 3)
colors_for_mesh = rgbs_resized[BG_MESH_T].reshape(-1, 3)

bg_vertices, bg_faces, bg_vertex_colors, bg_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud_for_mesh, 
    colors_for_mesh, 
    point_cloud_for_mesh[:,2] / fx 
)

# register background
bg_object_pose = Pose.from_translation(bg_vertices.mean(0))
bg_vertices = bg_object_pose.inverse().apply(bg_vertices)
poses.append(bg_object_pose)
object_library.add_object(bg_vertices, bg_faces, bg_vertex_colors)
all_poses = Pose.stack_poses(poses)


###################################
# Camera pose track
###################################
camera_pose_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict([
            ("camera_pose", Pose.identity()),
            *[(f"object_pose_{i}", all_poses[i]) for i in range(num_objects)],
            *[(f"object_{i}", i) for i in range(num_objects)],
            ("observed_rgb", rgbs_resized[BG_MESH_T]),
            ("observed_depth", xyzs[BG_MESH_T,...,2]),
        ])
    ),
    arguments
)
initial_camera_pose_trace = camera_pose_trace  # copy
initial_render = initial_camera_pose_trace.get_retval()[0][1]
# plt.imsave("initial.png", initial_render)

chain_cam_pose = []
chain_bg_pose = []
for T_observed_image in tqdm(range(START_CAM_T, START_JOINT_T+1, 1)):
    camera_pose_trace = b3d.update_choices_jit(camera_pose_trace, key,
        genjax.Pytree.const(["observed_rgb", "observed_depth"]),
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    camera_pose_trace,key = enumerative_proposal(camera_pose_trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)

    chain_cam_pose.append(camera_pose_trace['camera_pose'])
    chain_bg_pose.append(camera_pose_trace['object_pose_0'])
trace_at_seg_t = camera_pose_trace
bg_pose_at_seg_t = chain_bg_pose[-1]
cam_pose_at_seg_t = chain_cam_pose[-1]
rendered_rgb_at_seg_t = trace_at_seg_t.get_retval()[0][1]
rendered_depth_at_seg_t = trace_at_seg_t.get_retval()[1][1]

###################################
# SHOUT mesh acquisition
###################################
### determine a segmentation mask to construct a partial mesh from
cie_diff = b3d.ciede2000_err(b3d.rgb_to_lab(rgbs_resized[START_JOINT_T]), 
                                        b3d.rgb_to_lab(rendered_rgb_at_seg_t))
depth_diff = jnp.abs(rendered_depth_at_seg_t - xyzs[START_JOINT_T, ..., -1]) # TODO use CIEDE for inference as well
_pixels_to_cluster = (cie_diff > 10.0) & (depth_diff > depth_error) #TODO outlier mask

# acquire mesh from outlier pixels (TODO improve this method)
lb, num_feat = ndimage.label(_pixels_to_cluster)
_, frequency = jnp.unique(lb, return_counts=True)
sorted_indexes = jnp.argsort(frequency)[::-1]
object_feature_idx = sorted_indexes[1] # biggest cluster after background
pixels_to_cluster = _pixels_to_cluster * (lb == object_feature_idx)

plt.imsave("cluster.png", pixels_to_cluster)

point_cloud_for_obj_mesh = xyzs[START_JOINT_T][pixels_to_cluster]
colors_for_obj_mesh = rgbs_resized[START_JOINT_T][pixels_to_cluster]

# construct triangle mesh
shout_vertices, shout_faces, shout_vertex_colors, shout_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    point_cloud_for_obj_mesh, colors_for_obj_mesh, point_cloud_for_obj_mesh[:,2] / fx
)

# register Shout
shout_object_pose = Pose.from_translation(shout_vertices.mean(0))
shout_vertices = shout_object_pose.inverse().apply(shout_vertices)
object_library.add_object(shout_vertices, shout_faces, shout_vertex_colors)

###################################
# Camera pose + Object pose track
###################################
num_objects = 2  # bg and shout
poses = [bg_pose_at_seg_t, shout_object_pose] # pose at START_JOINT_T
all_poses = Pose.stack_poses(poses)

new_arguments = (jnp.empty((num_objects,)),) + arguments[1:]  # update number of objects
camera_and_shout_pose_trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        dict([
            ("camera_pose", cam_pose_at_seg_t),
            *[(f"object_pose_{i}", all_poses[i]) for i in range(num_objects)],
            *[(f"object_{i}", i) for i in range(num_objects)],  # 0 = bg 1 = shout
            ("observed_rgb", rgbs_resized[START_CAM_T]),
            ("observed_depth", xyzs[START_CAM_T,...,2]),
        ])
    ),
    new_arguments
)
chain_cam_pose_with_shout = []
chain_bg_pose_with_shout = []
chain_shout_pose = []
gt_rgb = []
render_rgb = []
for T_observed_image in tqdm(range(START_JOINT_T, END_T, 1)):
    camera_and_shout_pose_trace = b3d.update_choices_jit(camera_and_shout_pose_trace, key,
        genjax.Pytree.const(["observed_rgb", "observed_depth"]),
        rgbs_resized[T_observed_image],
        xyzs[T_observed_image,...,2]
    )
    camera_and_shout_pose_trace,key = enumerative_proposal(camera_and_shout_pose_trace, genjax.Pytree.const(["camera_pose"]), key, all_deltas)
    for i in [1]:  # ONLY enum over object pose (idx 1 not 0)
        camera_and_shout_pose_trace,key = enumerative_proposal(camera_and_shout_pose_trace, genjax.Pytree.const([f"object_pose_{i}"]), key, all_deltas)
    
    chain_cam_pose_with_shout.append(camera_and_shout_pose_trace['camera_pose'])
    chain_bg_pose_with_shout.append(camera_and_shout_pose_trace['object_pose_0'])
    chain_shout_pose.append(camera_and_shout_pose_trace['object_pose_1'])
    gt_rgb.append(rgbs_resized[T_observed_image])
    render_rgb.append(camera_and_shout_pose_trace.get_retval()[0][1])
    
    
trace_at_end_t = camera_and_shout_pose_trace
rendered_rgb_at_end_t = trace_at_end_t.get_retval()[0][1]
rendered_depth_at_end_t = trace_at_end_t.get_retval()[1][1]

plt.imsave("last_frame_render.png", rendered_rgb_at_end_t)
plt.imsave("last_frame_gt.png", gt_rgb[-1])

#############
# Viz results (TODO setup rerun)
#############

from PIL import Image
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

fig, axes = plt.subplots(1,3)
gif_ims = []
for gt, rend in tqdm(zip(gt_rgb[::5], render_rgb[::5])):
    axes[0].imshow(gt)
    axes[1].imshow(rend)
    axes[2].imshow((gt+rend)/2)
    gif_ims.append(fig2img(fig))

gif_ims[0].save("out.gif", save_all=True, append_images=gif_ims[1:], duration=100, loop=0)


from IPython import embed; embed()
