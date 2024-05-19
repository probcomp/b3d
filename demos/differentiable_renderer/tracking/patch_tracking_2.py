import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
#from b3d.utils import unproject_depth
import rerun as rr
import genjax
from tqdm import tqdm
import demos.differentiable_renderer.tracking.utils as utils
import demos.differentiable_renderer.tracking.model as m
import b3d.likelihoods as likelihoods
import b3d.differentiable_renderer as r

###
rendering = r
from demos.differentiable_renderer.tracking.model import uniform_pose, normalize

####

class ImageDistFromPixelDist(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    pixel_dist : any

    def _handle_arg_mapping(self, pixel_dist_args, width, height):
        print([a.shape for a in pixel_dist_args])
        print([isinstance(a, jnp.ndarray) and len(a.shape) > 0 and a.shape[0] > 1 for a in pixel_dist_args])
        do_map_args = [
            0 if (
                isinstance(arg, jnp.ndarray) and
                len(arg.shape) >= 2 and
                # arg.shape[:2] == (width, height)
                arg.shape[0] == width and arg.shape[1] == height
            ) else None
            for arg in pixel_dist_args
        ]
        flattened_args = [
            arg if (not do_map) else (
                arg.reshape(-1, *arg.shape[2:])
            )
            for (arg, do_map) in zip(pixel_dist_args, do_map_args)
        ]
        return do_map_args, flattened_args
    
    def sample(self, key, width, height, *pixel_dist_args):
        key, subkey = jax.random.split(key)
        
        do_map_args, flattened_args = self._handle_arg_mapping(pixel_dist_args, width, height)
        print(f"do_map_args: {do_map_args}")
        pixels = jax.vmap(
            lambda _, *args: self.pixel_dist.sample(subkey, *args),
            in_axes=(0, *do_map_args)
        )(jnp.arange(width*height), *flattened_args)
        return pixels.reshape((height, width))

    def logpdf(self, observed_image, width, height, *pixel_dist_args):
        if observed_image.shape != (height, width):
            return -jnp.inf

        do_map_args, flattened_args = self._handle_arg_mapping(pixel_dist_args, width, height)
        logpdfs = jax.vmap(
            lambda pixel, *args: self.pixel_dist.logpdf(pixel, *args),
            in_axes=(0, *do_map_args)
        )(observed_image.reshape(-1, observed_image.shape[-1]), *flattened_args)
        return logpdfs.sum()

class LaplaceMixtureDist(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, weights, attributes, scale):
        key, subkey = jax.random.split(key)
        choice = genjax.categorical.sample(subkey, jnp.log(weights))
        return likelihoods.laplace.sample(key, attributes[choice], scale)
    
    def logpdf(self, observed_value, weights, attributes, scale):
        logpdfs = jax.vmap(
            lambda attribute: self.laplace.logpdf(observed_value, attribute, scale),
            in_axes=(0,)
        )(weights, attributes)
        return jax.scipy.special.logsumexp(logpdfs + jnp.log(weights))

class UniformAndLaplaceMixtureDist(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def sample(self, key, weights, attributes, laplace_scale, minval, maxval):
        key, subkey = jax.random.split(key)
        choice = genjax.categorical.sample(subkey, jnp.log(weights))
        key, subkey = jax.random.split(key)
        uniform_value = genjax.uniform.sample(key, minval, maxval)
        laplace_value = likelihoods.laplace.sample(subkey, attributes[choice], laplace_scale)
        return jnp.where(choice == 0, uniform_value, laplace_value)
    
    def logpdf(self, observed_value, weights, attributes, laplace_scale, minval, maxval):
        uniform_logpdf = likelihoods.laplace.logpdf(observed_value, minval, maxval)
        laplace_logpdfs = jax.vmap(
            lambda attribute: likelihoods.laplace.logpdf(observed_value, attribute, laplace_scale)
        )(attributes)
        uniform_logpdf = jnp.log(weights[0] + 1e-5) + uniform_logpdf
        laplace_logpdfs = jnp.log(weights[1:] + 1e-5) + laplace_logpdfs
        return jax.scipy.special.logsumexp(jnp.concatenate([uniform_logpdf[None], laplace_logpdfs]))
uniform_laplace_mixture = UniformAndLaplaceMixtureDist()

class UniformLaplaceMixtureWithOutlier(genjax.ExactDensity,genjax.JAXGenerativeFunction):
    def _get_weights(weights, outlier_prob):
        # Increase the probability that pixel values are drawn from uniform by the outlier_prob.
        # (It will already be high if the pixel only hit the background.)
        weights = weights.at[:, :, 0].set(weights[:, :, 0] + outlier_prob)
        weights = jax.vmap(normalize, in_axes=(0,))(weights.reshape(-1, weights.shape[-1])).reshape(weights.shape)
        return weights

    def sample(self, key, weights, attributes, outlier_prob, laplace_scale, minval, maxval):
        return uniform_laplace_mixture.sample(
            key, self._get_weights(weights, outlier_prob),
            attributes, laplace_scale, minval, maxval
        )

    def logpdf(self, observed_value, weights, attributes, outlier_prob, laplace_scale, minval, maxval):
        return uniform_laplace_mixture.logpdf(
            observed_value, self._get_weights(weights, outlier_prob),
            attributes, laplace_scale, minval, maxval
        )
depth_pixel_model = UniformLaplaceMixtureWithOutlier()

####

def depth_model_singleobject_gl_factory(renderer,
                                  depth_noise_scale=0.07,
                                  outlier_prob = 0.05,
                                  hyperparams = rendering.DEFAULT_HYPERPARAMS,
                                  mindepth=0.,
                                  maxdepth=20.
                                ):
    @genjax.static_gen_fn
    def model(vertices_O, faces, vertex_colors):
        X_WO = uniform_pose(jnp.ones(3)*-10.0, jnp.ones(3)*10.0) @ "pose"
        X_WC = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"camera_pose"
        
        vertices_W = X_WO.apply(vertices_O)
        vertices_C = X_WC.inv().apply(vertices_W)

        weights, attributes = rendering.render_to_rgbd_dist_params(
            renderer, vertices_C, faces, vertex_colors, hyperparams
        )

        observed_depths = ImageDistFromPixelDist(depth_pixel_model)(
            renderer.width, renderer.height,
            weights, attributes[..., 3], outlier_prob, depth_noise_scale, mindepth, maxdepth
        ) @ "observed_depths"

        average_depths = rendering.dist_params_to_average(weights, attributes, jnp.zeros(1))
        return (observed_depths, average_depths)
    return model

########

# W = world frame; P = patch frame; C = camera frame

rr.init("track_test-4")
rr.connect("127.0.0.1:8812")

width = 100
height = 128
fx=64.0
fy=64.0
cx=64.0
cy=64.0
near=0.001
far=16.0

renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)

### Rotating box data ###
mesh_path = os.path.join(b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
mesh = trimesh.load(mesh_path)
cheezit_object_library = b3d.MeshLibrary.make_empty_library()
cheezit_object_library.add_trimesh(mesh)
rots = utils.vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
in_place_rots = b3d.Pose.from_matrix(rots)
cam_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.15, 0.15, 0.0]),
    jnp.array([0.0, 0.0, 0.0])
)
X_WC = cam_pose
compound_pose = X_WC.inv() @ in_place_rots
rgbs, depths = renderer.render_attribute_many(
    compound_pose[:,None,...],
    cheezit_object_library.vertices,
    cheezit_object_library.faces,
    jnp.array([[0, len(cheezit_object_library.faces)]]),
    cheezit_object_library.attributes
)
observed_rgbds = jnp.concatenate([rgbs, depths[...,None]], axis=-1)
xyzs_C = utils.unproject_depth_vec(depths, renderer)
xyzs_W = X_WC.apply(xyzs_C)
for t in range(rgbs.shape[0]):
    rr.set_time_sequence("frame", t)
    rr.log("/img/gt/rgb", rr.Image(rgbs[t, ...]))
    rr.log("/img/gt/depth", rr.Image(depths[t, ...]))
    rr.log("/3D/gt_pointcloud", rr.Points3D(
        positions=xyzs_W[t].reshape(-1,3),
        colors=rgbs[t].reshape(-1,3),
        radii = 0.001*np.ones(xyzs_W[t].reshape(-1,3).shape[0]))
    )
rr.log("/3D/camera",
       rr.Pinhole(
           focal_length=renderer.fx,
           width=renderer.width,
           height=renderer.height,
           principal_point=jnp.array([renderer.cx, renderer.cy]),
        ), timeless=True
    )
rr.log("/3D/camera", rr.Transform3D(translation=cam_pose.pos, mat3x3=cam_pose.rot.as_matrix()), timeless=True)

### Get patch ###
center_x, center_y = 40, 45
del_pix = 5
patch_points_C = jax.lax.dynamic_slice(xyzs_C[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
patch_vertices_C, patch_faces, patch_vertex_colors, patch_face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    patch_points_C, patch_rgbs, patch_points_C[:,2] / fx * 2.0
)
X_CP = Pose.from_translation(patch_vertices_C.mean(0))
patch_vertices_P = X_CP.inv().apply(patch_vertices_C)
patch_vertices_W = X_WC.apply(patch_vertices_C)
X_WP = X_WC @ X_CP

### Set up diff rend ###
hyperparams = r.DifferentiableRendererHyperparams(
    3, 1e-5, 1e-2, -1
)

###


model = depth_model_singleobject_gl_factory(
    renderer,
    hyperparams=hyperparams,
    depth_noise_scale=1e-2,
    mindepth=-1.0,
    maxdepth=1.0,
    outlier_prob=0.05
)
tr = model.simulate(jax.random.PRNGKey(0), (patch_vertices_P, patch_faces, patch_vertex_colors))

[
    0 if (
        isinstance(arg, jnp.ndarray) and
        len(arg.shape) >= 2 and
        arg.shape[0] == width and arg
    ) else None
    for arg in args
]

#                          b3d.Pose    (H, W, 4) array
def pose_to_trace_and_weight(pose, observed_depth):
    vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_depth)))
    constraints = genjax.choice_map({
                             "pose": pose,
                             "camera_pose": X_WC,
                             "observed_depth": observed_depth
                         })
    trace, weight = model.importance(jax.random.PRNGKey(0), constraints, (patch_vertices_P, patch_faces, patch_vertex_colors))
    return trace, weight

pose_to_trace_and_weight(X_WP, observed_depths[0])

@jax.jit
def pos_quat_to_trace_and_weight(position, quaternion, observed_depth):
    pose = b3d.Pose(position, quaternion)
    return pose_to_trace_and_weight(pose, observed_depth)

def pos_quat_to_weight(position, quaternion, observed_depth):
    return pos_quat_to_trace_and_weight(position, quaternion, observed_depth)[1]

pos_quat_grad_jitted = jax.grad(pos_quat_to_weight, argnums=(0, 1))
step_pos = 1e-7
step_quat = 1e-7
CLIP_STEP = 0.001
CLIP_QUAT = 0.001
@jax.jit
def step_pose(position, quaternion, observed_depth):
    gp, gq = pos_quat_grad_jitted(position, quaternion, observed_depth)
    p_inc = step_pos * gp
    p_inc = jnp.where(jnp.linalg.norm(p_inc) > CLIP_STEP, p_inc / jnp.linalg.norm(p_inc) * CLIP_STEP, p_inc)
    q_inc = step_quat * gq
    q_inc = jnp.where(jnp.linalg.norm(q_inc) > CLIP_QUAT, q_inc / jnp.linalg.norm(q_inc) * CLIP_QUAT, q_inc)    
    position = position + p_inc
    quaternion = quaternion + q_inc
    quaternion = quaternion / jnp.linalg.norm(quaternion)
    return position, quaternion

position = X_WP.pos + jnp.array([0.0001, 0.002, -0.0031])
quaternion = X_WP._quaternion
STEPS_PER_FRAME=50
key = jax.random.PRNGKey(12)
observed_depths = observed_rgbds[...,3]
for fr in tqdm(range(observed_depths.shape[0])):
    for step in tqdm(range(STEPS_PER_FRAME)):
        if step > 0:
            # position, quaternion = step_pose(position, quaternion, observed_rgbds[fr])
            key, subkey = jax.random.split(key)
            (position, quaternion) = step_pose(position, quaternion, observed_depths[fr])
        trace, weight = pos_quat_to_trace_and_weight(position, quaternion, observed_depths[fr])
        rr.set_time_sequence("gradient_ascent_step1", fr*STEPS_PER_FRAME + step)
        m.rr_viz_trace(trace, renderer)
        rr.log("/trace/logpdf", rr.Scalar(trace.get_score()))

    rr.set_time_sequence("frames_gradient_tracking", fr)
    m.rr_viz_trace(trace, renderer)
