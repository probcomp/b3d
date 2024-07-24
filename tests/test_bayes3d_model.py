import b3d
import b3d.bayes3d as bayes3d
import genjax
import jax
import jax.numpy as jnp
from b3d import Pose
from b3d.bayes3d.model import model_multiobject_gl_factory
from genjax import ChoiceMapBuilder as C


class TestGroup:
    key = jax.random.PRNGKey(0)

    object_library = bayes3d.MeshLibrary.make_empty_library()
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )

    @genjax.gen
    def object_gf(object_library):
<<<<<<< HEAD
        object_identity = (
=======
        (
>>>>>>> main
            b3d.modeling_utils.uniform_discrete(
                jnp.arange(-1, object_library.get_num_objects())
            )
            @ "id"
        )
<<<<<<< HEAD
        object_pose = (
=======
        (
>>>>>>> main
            b3d.modeling_utils.uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0)
            @ "pose"
        )

    trace = object_gf.simulate(key, (object_library,))

    object_library = bayes3d.MeshLibrary.make_empty_library()
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )
    object_library.add_object(
        jnp.zeros((100, 3)), jnp.zeros((10, 3), dtype=jnp.int32), jnp.zeros((100, 3))
    )

    model_args = bayes3d.ModelArgs(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

    def test_importance(self, renderer):
        model = model_multiobject_gl_factory(renderer)
        trace, _ = model.importance(
            jax.random.PRNGKey(0),
            C.d(
                {
                    "camera_pose": Pose.identity(),
                    "object_pose_0": Pose.identity(),
                }
            ),
            (jnp.arange(3), self.model_args, self.object_library),
        )
        identity_pose = Pose.identity()
        assert jnp.allclose(
            trace.get_choices()["camera_pose"].position, identity_pose.position
        )
        assert jnp.allclose(
            trace.get_choices()["camera_pose"].quaternion, identity_pose.quaternion
        )
        assert jnp.allclose(
            trace.get_choices()["object_pose_0"].position, identity_pose.position
        )
        assert jnp.allclose(
            trace.get_choices()["object_pose_0"].quaternion, identity_pose.quaternion
        )
