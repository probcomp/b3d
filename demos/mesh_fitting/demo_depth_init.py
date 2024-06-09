import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.differentiable_renderer
import b3d.tessellation as t
import b3d.utils as u
import os
import rerun as rr
import optax
from tqdm import tqdm

vcm = genjax.vector_choice_map
c = genjax.choice

import demos.mesh_fitting.model as m
import demos.mesh_fitting.utils as u

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
    # "assets/potted_plant.video_input.npz"
)
video_input = b3d.VideoInput.load(path)
(vertices_3D, faces, triangle_rgbds, renderer, rgbs) = u.initialize_mesh_using_depth(video_input)
triangle_colors = triangle_rgbds[:, :3]

camera_poses = [
    b3d.Pose(video_input.camera_positions[::4][t], video_input.camera_quaternions[::4][t])
    for t in range(rgbs.shape[0])
]
initial_mesh = (camera_poses[0].apply(vertices_3D), faces, triangle_colors)

### Generate + log initial trace ###
model = m.get_rgb_only_model(renderer, initial_mesh)

rr.init("depth_mesh_init-4")
rr.connect("127.0.0.1:8812")

key = jax.random.PRNGKey(0)
trace, weight = jax.jit(model.importance)(
    key, genjax.choice_map({
        "vertices": initial_mesh[0], "faces": initial_mesh[1], "face_colors": initial_mesh[2],
        "camera_poses": vcm(c(b3d.Pose.stack_poses([camera_poses[t] for t in [0]]))),
        "observed_rgbs": vcm(genjax.choice_map({"observed_rgb": rgbs[[0], ...]}))
    }), ())

m.rr_log_trace(trace, renderer, "mytrace", [0], [0])

### 

frames = [0]
@jax.jit
def importance_from_vertices_colors(vertices, colors):
    return model.importance(
        key, genjax.choice_map({
            "vertices": vertices, "faces": initial_mesh[1], "face_colors": colors,
            "camera_poses": vcm(c(b3d.Pose.stack_poses([camera_poses[t] for t in frames]))),
            "observed_rgbs": vcm(genjax.choice_map({"observed_rgb": rgbs[frames, ...]}))
        }), ())

def vertices_colors_to_score(vertices, colors):
    trace, weight = importance_from_vertices_colors(vertices, colors)
    return weight
grad_jitted = jax.jit(jax.grad(vertices_colors_to_score, argnums=(0, 1)))
value_and_grad_jitted = jax.jit(jax.value_and_grad(vertices_colors_to_score, argnums=(0, 1)))
