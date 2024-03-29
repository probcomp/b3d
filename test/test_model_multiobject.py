import b3d
from b3d.model import model_gl_factory, model_multiobject_gl_factory
from b3d import Pose
import jax
import genjax
import jax.numpy as jnp
import trimesh
import rerun as rr

rr.init("demo.py")
rr.connect("127.0.0.1:8812")

from pathlib import Path
mesh_path = Path(b3d.__file__).parents[1] / "assets/006_mustard_bottle/textured_simple.obj"
mesh = trimesh.load(mesh_path)

vertices = jnp.array(mesh.vertices) * 20.0
vertices = vertices - vertices.mean(0)
faces = jnp.array(mesh.faces)
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0
ranges = jnp.array([[0, len(faces)]])

object_library = b3d.model.MeshLibrary.make_empty_library()
object_library.add_object(vertices, faces, vertex_colors)
object_library.add_object(vertices, faces, vertex_colors)


width=100
height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    width, height, fx, fy, cx, cy, near, far
)


model = model_multiobject_gl_factory(renderer)


importance_jit = jax.jit(model.importance)

trace, _ = importance_jit(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose_0": Pose.from_position_and_target(jnp.array([2.4, 0.4, 0.0]), jnp.zeros(3)).inv(),
            "object_pose_1": Pose.from_position_and_target(jnp.array([2.4, 0.4, 0.0]), jnp.ones(3)).inv(),
        
        }
    ),
    (
        jnp.arange(3),
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        object_library
    ),
)

b3d.rerun_visualize_trace_t(trace, 0)