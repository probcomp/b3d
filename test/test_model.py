import b3d
from b3d.model import model_multiobject_gl_factory
from b3d import Pose
import jax
import genjax
import jax.numpy as jnp


class TestGroup:

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

    key = jax.random.PRNGKey(0)

    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))



    @genjax.static_gen_fn
    def object_gf(object_library):
        object_identity = b3d.uniform_discrete(jnp.arange(-1, object_library.get_num_objects())) @ f"id"
        object_pose = b3d.uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"pose"

    trace = object_gf.simulate(key, (object_library,))

    @genjax.static_gen_fn
    def model(dummy_num_objects):
        genjax.map_


    model = model_multiobject_gl_factory(renderer)

    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))
    object_library.add_object(jnp.zeros((100,3)), jnp.zeros((10,3),dtype=jnp.int32), jnp.zeros((100,3)))

    model_args = b3d.ModelArgs(
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    )



    def test_importance(self):
        model = self.model
        trace, _ = model.importance(
            jax.random.PRNGKey(0),
            genjax.choice_map(
                {
                    "camera_pose": Pose.identity(),
                    "object_pose_0": Pose.identity(),
                }
            ),
            (
                jnp.arange(3),
                self.model_args,
                self.object_library
            ),
        )
        identity_pose = Pose.identity()
        assert jnp.allclose(trace["camera_pose"].position, identity_pose.position)
        assert jnp.allclose(trace["camera_pose"].quaternion, identity_pose.quaternion)
        assert jnp.allclose(trace["object_pose_0"].position, identity_pose.position)
        assert jnp.allclose(trace["object_pose_0"].quaternion, identity_pose.quaternion)
