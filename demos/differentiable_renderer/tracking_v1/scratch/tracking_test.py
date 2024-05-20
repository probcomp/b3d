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
import demos.differentiable_renderer.tracking_v1.utils as utils
import demos.differentiable_renderer.tracking_v1.model as m

rr.init("tracking_test_4")
rr.connect("127.0.0.1:8812")

width=128
height=128
fx=64.0
fy=64.0
cx=64.0
cy=64.0
near=0.001
far=16.0

renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)

mesh_path = os.path.join(b3d.get_root_path(),
    "assets/shared_data_bucket/ycb_video_models/models/003_cracker_box/textured_simple.obj")
mesh = trimesh.load(mesh_path)

object_library = b3d.MeshLibrary.make_empty_library()
object_library.add_trimesh(mesh)

rots = utils.vec_transform_axis_angle(jnp.array([0,0,1]), jnp.linspace(jnp.pi/4, 3*jnp.pi/4, 30))
in_place_rots = b3d.Pose.from_matrix(rots)

cam_pose = b3d.Pose.from_position_and_target(
    jnp.array([0.15, 0.15, 0.0]),
    jnp.array([0.0, 0.0, 0.0])
)
compound_pose = cam_pose.inv() @ in_place_rots
rgbs, depths = renderer.render_attribute_many(
    compound_pose[:,None,...],
    object_library.vertices,
    object_library.faces,
    jnp.array([[0, len(object_library.faces)]]),
    object_library.attributes
)
xyzs = utils.unproject_depth_vec(depths, renderer)
for t in range(rgbs.shape[0]):
    rr.set_time_sequence("frame", t)
    rr.log("/gt/rgb", rr.Image(rgbs[t, ...]))
    rr.log("/gt/depth", rr.Image(depths[t, ...]))
    rr.log("/gt_pointcloud", rr.Points3D(
        positions=xyzs[t].reshape(-1,3),
        colors=rgbs[t].reshape(-1,3),
        radii = 0.001*np.ones(xyzs[t].reshape(-1,3).shape[0]))
    )

###
center_x, center_y = 40, 45
del_pix = 5
local_points = jax.lax.dynamic_slice(xyzs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
local_rgbs = jax.lax.dynamic_slice(rgbs[0], (center_x-del_pix,center_y-del_pix,0), (2*del_pix,2*del_pix,3)).reshape(-1,3)
vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
    local_points, local_rgbs, local_points[:,2] / fx * 2.0
)

object_pose = Pose.from_translation(vertices.mean(0))
vertices = object_pose.inverse().apply(vertices)
object_library.add_object(vertices, faces, vertex_colors)

rr.set_time_sequence("frame", 0)
rr.log("mesh", rr.Mesh3D(
    vertex_positions=object_pose.apply(vertices),
    indices=faces,
    vertex_colors=vertex_colors
))

vertices = object_pose.apply(vertices)
vertex_rgbs = vertex_colors
vertex_attributes = vertex_rgbds

import b3d.differentiable_renderer as r
hyperparams = r.DifferentiableRendererHyperparams(
        3, 1e-8, 0.25, 0.
    )
rr.log("vertices", rr.Points3D(positions=object_pose.apply(vertices), colors=jnp.ones_like(vertices), radii=0.001))
rr.log("trace_camera", rr.Pinhole(focal_length=renderer.fx, width=renderer.width, height=renderer.height))
rendered = r.render_to_average_rgbd(
    renderer,
    # object_pose.apply(vertices),
    vertices,
    faces,
    vertex_colors,
    background_attribute=jnp.array([0.1, 0.1, 0.1, 10.]),
    hyperparams=hyperparams
)

rr.log("/patch/rendered_rgb", rr.Image(rendered[:, :, :3]), timeless=True)
# rr.log("/rendered/depth", rr.DepthImage(rendered[:, :, 3]), timeless=True)

compound_pose = cam_pose.inv() @ in_place_rots
rgbs, depths = renderer.render_attribute(
    Pose.identity()[None, ...],
    vertices,q
    faces,
    jnp.array([[0, len(faces)]]),
    vertex_colors
)
rr.log("/gt2/rgb", rr.Image(rgbs))

# rgbs, depths = renderer.render_attribute_many(
#     compound_pose[:,None,...],
#     object_library.vertices,
#     object_library.faces,
#     jnp.array([[0, len(object_library.faces)]]),
#     object_library.attributes
# )

ij = jnp.array([40, 44])




importlib.reload(m)
hyperparams = b3d.differentiable_renderer.DifferentiableRendererHyperparams(
        3, 1e-5, 1e-3, -1
    )
model = m.model_singleobject_gl_factory(renderer)
key = jax.random.PRNGKey(1)
trace = model.simulate(key, (vertices, faces, vertex_colors))

observed_rgbd = jnp.concatenate([rgbs[0], depths[0, :, :, None]], axis=-1) 
vcm = genjax.vector_choice_map(genjax.vector_choice_map(genjax.choice(observed_rgbd)))
constraints = genjax.choice_map({
                             "pose": object_pose,
                             "camera_pose": cam_pose,
                             "observed_rgbd": vcm
                         })
trace, weight = model.importance(key, constraints, (vertices, faces, vertex_colors))
m.rr_viz_trace(trace, renderer)