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

import demos.mesh_fitting.model as m

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
    # "assets/potted_plant.video_input.npz"
)
video_input = b3d.VideoInput.load(path)

width, height, fx, fy, cx, cy, near, far = jnp.array(video_input.camera_intrinsics_depth)
width, height = int(width), int(height)
fx, fy, cx, cy, near, far = (float(fx), float(fy), float(cx), float(cy), float(near), float(far),)
renderer = b3d.Renderer(width, height, fx, fy, cx, cy, near, far)
rgbs_full_resolution = video_input.rgb[::4] / 255.0
rgbs = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs_full_resolution, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)
depths = video_input.xyz[::4][:, :, :, 3]
rgbds = jnp.concatenate([rgbs, depths[..., None]], axis=-1)

vertices_2D, faces, triangle_rgbds = t.generate_tessellated_2D_mesh_from_rgb_image(rgbds[0], scaledown=30)

MAX_N_FACES = 7
def get_faces_for_vertex(i):
    return jnp.where(faces == i, size=MAX_N_FACES, fill_value=-1)[0]
vertex_to_faces = jax.vmap(get_faces_for_vertex)(jnp.arange(vertices_2D.shape[0]))
# Check we had 1 more padding than we needed, for each vertex
assert jnp.all(jnp.any(vertex_to_faces == -1, axis=1))

def get_vertex_depth(v):
    face_indices = vertex_to_faces[v]
    face_indices_safe = jnp.where(face_indices == -1, 0, face_indices)
    depths = jnp.where(face_indices != -1, triangle_rgbds[face_indices_safe, 3], 0.)
    n_valid = jnp.sum(depths != 0)
    return jnp.sum(depths) # / n_valid
vertex_depths = jax.vmap(get_vertex_depth)(jnp.arange(vertices_2D.shape[0]))

vertices_3D = jnp.hstack(
    ((vertices_2D  - jnp.array([cx, cy])) * vertex_depths[:, None] / jnp.array([fx, fy]) ,
    vertex_depths[:, None])
)
triangle_colors = triangle_rgbds[:, :3]

rr.init("depth_mesh_init3")
rr.connect("127.0.0.1:8812")
# v_, f_, vc_ = b3d.utils.triangle_color_mesh_to_vertex_color_mesh(vertices_3D, faces, triangle_colors)
# rr.log(f"/3D/mesh", rr.Mesh3D(
#     vertex_positions=v_, indices=f_, vertex_colors=vc_
# ))

camera_poses = [
    b3d.Pose(video_input.camera_positions[::4][t], video_input.camera_quaternions[::4][t])
    for t in range(rgbs.shape[0])
]

initial_mesh = (camera_poses[0].apply(vertices_3D), faces, triangle_colors)


mindepth, maxdepth = -1000., 1000.
color_scale = 0.05
outlier_prob = 0.05
def normalize(x):
    return x / jnp.sum(x)
likelihood = b3d.likelihoods.ArgMap(
    b3d.likelihoods.get_uniform_multilaplace_rgbonly_image_dist_with_fixed_params(
        renderer.height, renderer.width, color_scale
    ),
    lambda weights, attributes:(
        normalize(jnp.concatenate([weights[:1] + outlier_prob, weights[1:]])),
        attributes
    )
)
hyperparams = b3d.differentiable_renderer.DifferentiableRendererHyperparams(3, 1e-5, 1e-2, -1)
model = m.model_factory(
    renderer, likelihood, hyperparams, mindepth, maxdepth, 1,
    initial_mesh[0].shape[0], initial_mesh[1].shape[0]
)


key = jax.random.PRNGKey(0)
trace, weight = jax.jit(model.importance)(
    key,
    genjax.choice_map({
        "vertices": initial_mesh[0],
        "faces": initial_mesh[1],
        "face_colors": initial_mesh[2],
        "camera_poses": genjax.vector_choice_map(
            genjax.choice( b3d.Pose.stack_poses([camera_poses[t] for t in [0]]))
        ),
        "observed_rgbs": genjax.vector_choice_map(genjax.choice_map({"observed_rgb": rgbs[[0], ...]}))
    }),
    ()
)

m.rr_log_trace(trace, renderer, "trace2", [0], [0])

frames = [0]
@jax.jit
def importance_from_vertices_colors(vertices, colors):
    trace, weight = jax.jit(model.importance)(
        key,
        genjax.choice_map({
            "vertices": vertices,
            "faces": initial_mesh[1],
            "face_colors": colors,
            "camera_poses": genjax.vector_choice_map(
                genjax.choice( b3d.Pose.stack_poses([camera_poses[t] for t in frames]))
            ),
            "observed_rgbs": genjax.vector_choice_map(genjax.choice_map({"observed_rgb": rgbs[frames, ...]}))
        }),
        ()
    )
    return trace, weight

def vertices_colors_to_score(vertices, colors):
    trace, weight = importance_from_vertices_colors(vertices, colors)
    return weight
grad_jitted = jax.jit(jax.grad(vertices_colors_to_score, argnums=(0, 1)))
value_and_grad_jitted = jax.jit(jax.value_and_grad(vertices_colors_to_score, argnums=(0, 1)))


hessian = jax.jit(jax.hessian(vertices_colors_to_score, argnums=(0, 1)))

def vc_to_s(vc):
    vertices = vc[:(initial_mesh[0].shape[0]), :]
    colors = vc[(initial_mesh[0].shape[0]):, :]
    return vertices_colors_to_score(vertices, colors)

vc = jnp.vstack((initial_mesh[0], initial_mesh[2]))

jax.grad(vc_to_s)(vc)
jax.hessian(vertices_colors_to_score, argnums=0)(initial_mesh[0], initial_mesh[2])

jax.jacrev(jax.jacrev(vc_to_s))(vc)



jax.jacrev(vc_to_s)(vc)

jax.hessian(vc_to_s)(vc)

v = jax.jacrev(jax.jacfwd(vc_to_s))(vc)

# h = jax.hessian(vertices_colors_to_score, argnums=(0, 1))(initial_mesh[0], initial_mesh[2])

# g = grad_jitted(initial_mesh[0], initial_mesh[2])

with jax.profiler.trace("/tmp/tensorboard"):
  # Run the operations to be profiled
  g = value_and_grad_jitted(initial_mesh[0] + 1.0, initial_mesh[2])
  g[0].block_until_ready()
  g[1][0].block_until_ready()