import b3d
from b3d.model import model_gl_factory, model_multiobject_gl_factory
from b3d import Pose
import jax
import genjax
import jax.numpy as jnp

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

model = model_gl_factory(renderer)

trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            "camera_pose": Pose.identity(),
            "object_pose": Pose.identity(),
        }
    ),
    (
        jnp.zeros((100,3)), jnp.zeros((100,3),dtype=jnp.int32), jnp.zeros((100,3)),
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ),
)

model = model_multiobject_gl_factory(renderer)

object_library = b3d.model.MeshLibrary()
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))


trace, _ = model.importance(
    jax.random.PRNGKey(0),
    genjax.choice_map(
        {
            # "camera_pose": Pose.identity(),
            # "object_pose": Pose.identity(),
        }
    ),
    (
        jnp.arange(3),
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        object_library
    ),
)
