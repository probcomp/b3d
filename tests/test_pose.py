import unittest

import jax
import jax.numpy as jnp
import numpy as np
from b3d.pose import Pose, camera_from_position_and_target
from jax.scipy.spatial.transform import Rotation as Rot


def keysplit(key, *ns):
    if len(ns) == 0:
        return jax.random.split(key, 1)[0]
    elif len(ns) == 1:
        (n,) = ns
        if n == 1:
            return keysplit(key)
        else:
            return jax.random.split(key, ns[0])
    else:
        keys = []
        for n in ns:
            keys.append(keysplit(key, n))
        return keys


class PoseTests(unittest.TestCase):
    key = jax.random.PRNGKey(np.random.randint(0, 10_000))

    def test_pose_properties(self):
        keys = keysplit(self.key, 2)
        x = jax.random.normal(keys[0], (3,))
        q = jax.random.normal(keys[1], (4,))
        p = Pose(x, q)
        self.assertTrue(jnp.allclose(p.pos, x))
        self.assertTrue(jnp.allclose(p.xyzw, q))
        self.assertTrue(jnp.allclose(jnp.concatenate(p.flat), jnp.concatenate((x, q))))

    def test_pose_inv(self):
        key = self.key

        N = 50
        keys = keysplit(key, 3)
        xs = jax.random.normal(keys[0], (N, 3))
        qs = jax.random.normal(keys[1], (N, 4))
        vs = jax.random.normal(keys[2], (N, 3))
        ps = Pose(xs, qs)

        self.assertTrue(jnp.allclose(ps.inv().apply(ps.apply(vs)), vs, atol=1e-5))

        def choose_good_quat(q):
            """
            If the real part of the quaternion is negative,
            return the antipodal quaternion,
            which represents the same rotation.

            Recall that SO(3) is isomorphic to  S^3/x~-x and
            also to D^3/~ where x~-x for x in S^2 = \partial D^3.
            """
            # TODO: choose good representative if q[3]==0 there is still ambiguity.
            return jnp.where(q[..., [3]] == 0, q, jnp.sign(q[..., [3]]) * q)

        # Reality check if q and -q describe the same rotation.
        self.assertTrue(
            jnp.allclose(
                Rot.from_quat(qs).as_matrix(),
                Rot.from_quat(jnp.array(-1) * qs).as_matrix(),
                atol=1e-4,
            )
        )

        computed = ps @ ps.inv()
        expected = Pose(
            jnp.zeros((N, 3)), jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (N, 1))
        )

        self.assertTrue(
            jnp.allclose(computed.pos, expected.pos, atol=1e-4),
            computed.pos - expected.pos,
        )

        self.assertTrue(
            jnp.allclose(choose_good_quat(computed.xyzw), expected.xyzw, atol=1e-4),
            computed.xyzw - expected.xyzw,
        )

    def test_compose(self):
        key = self.key
        keys = keysplit(key, 4)
        a = Pose(jax.random.uniform(keys[0], (3,)), jax.random.uniform(keys[1], (4,)))
        b = Pose(jax.random.uniform(keys[2], (3,)), jax.random.uniform(keys[3], (4,)))

        mat_a = a.as_matrix()
        mat_b = b.as_matrix()

        mat_c = mat_a @ mat_b
        c = a @ b
        self.assertTrue(jnp.allclose(c.as_matrix(), mat_c))

    def test_apply(self):
        key = self.key
        N = 100
        points = jax.random.uniform(key, (N, 3))
        pose = Pose(jax.random.uniform(key, (3,)), jax.random.uniform(key, (4,)))

        expected = (
            pose.as_matrix() @ jnp.concatenate([points, jnp.ones((100, 1))], axis=1).T
        ).T[:, :3]
        self.assertTrue(jnp.allclose(pose.apply(points), expected, atol=1e-3))
        self.assertTrue(jnp.allclose(pose.apply(points), pose(points)))

        identity = Pose.identity()
        self.assertTrue(jnp.allclose(identity.apply(points), points))

    def test_camera_pose_from_position_target(self):
        keys = keysplit(self.key, 2)
        pos = jax.random.uniform(keys[0], (3,))
        target = jax.random.uniform(keys[1], (3,))
        pose = camera_from_position_and_target(pos, target)
        self.assertTrue(jnp.allclose(pose.pos, pos))

        camera_vector = Rot.from_quat(pose.xyzw).apply(jnp.array([0.0, 0.0, 1.0]))

        self.assertTrue(
            jnp.allclose(camera_vector, (target - pos) / jnp.linalg.norm(target - pos))
        )

    def test_grad(self):
        key = self.key
        N = 100
        random_points = jax.random.uniform(key, (N, 3))
        pose = Pose(jax.random.uniform(key, (3,)), jax.random.uniform(key, (4,)))
        transformed_points = pose.apply(random_points)

        def loss_func(pose):
            return jnp.abs(pose.apply(random_points) - transformed_points).mean()

        keys = keysplit(key, 2)
        pose = Pose(
            jax.random.uniform(keys[0], (3,)), jax.random.uniform(keys[1], (4,))
        )
        loss = loss_func(pose)
        for _ in range(10):
            loss, grad = jax.value_and_grad(loss_func)(pose)
            pose = Pose(pose.pos - 0.001 * grad.pos, pose.xyzw - 0.001 * grad.xyzw)
        loss_end = loss_func(pose)
        self.assertTrue(loss_end < loss)

    def test_aliases(self):
        keys = keysplit(self.key, 2)
        pos = jax.random.uniform(keys[0], (3,))
        target = jax.random.uniform(keys[1], (3,))
        pose = camera_from_position_and_target(pos, target)
        self.assertTrue(jnp.allclose(pose.pos, pose.position))
        self.assertTrue(jnp.allclose(pose.quat, pose.xyzw))
        self.assertTrue(jnp.allclose(pose.quaternion, pose.xyzw))

    def test_loop_termination(self):
        keys = keysplit(self.key, 2)
        poses = Pose(
            jax.random.uniform(keys[0], (10, 3)), jax.random.uniform(keys[1], (10, 4))
        )

        def sum(poses):
            sum = jnp.zeros(3)
            for pose in poses:
                sum = sum.at[0:3].add(pose.pos[0:3])
            return sum

        sum_jit = jax.jit(sum)

        self.assertTrue(jnp.allclose(jnp.sum(poses.pos, axis=0), sum_jit(poses)))

    def test_multiindexing(self):
        identity_pose = Pose.identity()
        identity_pose_multiple_dimensions = identity_pose[None, None, ...]
        assert identity_pose_multiple_dimensions.shape == (
            1,
            1,
        ), identity_pose_multiple_dimensions.shape
