import b3d
import b3d.bayes3d as bayes3d
from b3d.bayes3d.model import model_multiobject_gl_factory
from b3d import Pose
import jax
import genjax
import jax.numpy as jnp
import pytest


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

    @genjax.static_gen_fn
    def object_gf(object_library):
        object_identity = (
            b3d.modeling_utils.uniform_discrete(jnp.arange(-1, object_library.get_num_objects()))
            @ f"id"
        )
        object_pose = (
            b3d.modeling_utils.uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ f"pose"
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
            genjax.choice_map(
                {
                    "camera_pose": Pose.identity(),
                    "object_pose_0": Pose.identity(),
                }
            ),
            (jnp.arange(3), self.model_args, self.object_library),
        )
        identity_pose = Pose.identity()
        assert jnp.allclose(trace["camera_pose"].position, identity_pose.position)
        assert jnp.allclose(trace["camera_pose"].quaternion, identity_pose.quaternion)
        assert jnp.allclose(trace["object_pose_0"].position, identity_pose.position)
        assert jnp.allclose(trace["object_pose_0"].quaternion, identity_pose.quaternion)
