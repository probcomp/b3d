from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jax import vmap
import jax
from jax.scipy.spatial.transform import Rotation as Rot
from typing import NamedTuple
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp
from warnings import warn
import b3d
from typing import Any, NamedTuple, TypeAlias
import numpy as np
import cv2

Array: TypeAlias = jax.Array
Float: TypeAlias = Array
Int: TypeAlias = Array
Quaternion: TypeAlias = Array


def multiply_quats(q1, q2):
    return (Rot.from_quat(q1) * Rot.from_quat(q2)).as_quat()


def multiply_quat_and_vec(q, vs):
    return Rot.from_quat(q).apply(vs)


def sample_uniform_pose(key, low, high):
    keys = jax.random.split(key, 2)
    pos = jax.random.uniform(keys[0], (3,)) * (high - low) + low
    quat = jax.random.normal(keys[1], (4,))
    quat = quat / jnp.linalg.norm(quat)
    return Pose(pos, quat)


def logpdf_uniform_pose(pose, low, high):
    position = pose.position
    valid = (low <= position) & (position <= high)
    position_score = jnp.log((valid * 1.0) * (jnp.ones_like(position) / (high - low)))
    return position_score.sum() + jnp.pi**2


def sample_gaussian_vmf_pose(key, mean_pose, std, concentration):
    """
    Samples poses from the product of a diagonal normal distribution (for position) and
    a generalized von Mises-Fisher distribution (for quaternion).

    Note:
    One can view the von Mises–Fisher distribution over the n-sphere
    as the restriction of the normal distribution on R^{n+1}
    to the n-sphere. From this viewpoint the concentration is
    approximateley the inverse of the variance.

    See:
    > https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution#Relation_to_normal_distribution
    """
    translation = tfp.distributions.MultivariateNormalDiag(
        jnp.zeros(3), jnp.ones(3) * std
    ).sample(seed=key)
    key = jax.random.split(key, 1)[0]
    quat = tfp.distributions.VonMisesFisher(
        jnp.array([0.0, 0.0, 0.0, 1.0]), concentration
    ).sample(seed=key)
    return mean_pose @ Pose(translation, quat)


def camera_from_position_and_target(
    position, target=jnp.array([0.0, 0.0, 0.0]), up=jnp.array([0.0, 0.0, 1.0])
):
    """
    Create a camera pose at `position` with the camera-z-axis pointint at `target`.
    Recall that in world coordinates we assume z-axis is up.

    Args:
        `position`: 3D position vector of the camera
        `target`: 3D position vector of the point to look at
        `up`: 3D vector pointing up.
    """
    z = target - position
    z = z / jnp.linalg.norm(z)

    x = jnp.cross(z, up)
    x = x / jnp.linalg.norm(x)

    y = jnp.cross(z, x)
    y = y / jnp.linalg.norm(y)

    rotation_matrix = jnp.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
    return Pose(position, Rot.from_matrix(rotation_matrix).as_quat())


@register_pytree_node_class
class Pose:
    """Pose class with positions and quaternions representing rotation."""

    def __init__(self, position, quaternion):
        """
        3D rigid transformation

        position: 3D translation vector
        quaternion: 4D quaternion in xyzw
        """
        self._position = position
        self._quaternion = quaternion

    identity_quaternion = jnp.array([0.0, 0.0, 0.0, 1.0])

    @property
    def unit_quaternion(self):
        raise Warning(
            "Use `identity_quaternion` instead, a unit quaternion is any quat with norm 1!"
        )
        return identity_quaternion

    @property
    def pos(self):
        return self._position

    position = pos

    @property
    def xyzw(self):
        return self._quaternion

    quat = xyzw
    quaternion = xyzw

    @property
    def wxyz(self):
        return jnp.concatenate(
            [self.quaternion[..., 3:4], self.quaternion[..., :3]], axis=-1
        )

    @property
    def rot(self):
        return Rot.from_quat(self.xyzw)

    def normalize(self):
        quat = self.quat / jnp.linalg.norm(self.quat, axis=-1, keepdims=True)
        return Pose(self.pos, quat)

    def quat_in_upper_hemisphere(self):
        quat = self.quat / jnp.linalg.norm(self.quat, axis=-1, keepdims=True)
        quat = jnp.sign(quat[..., [3]]) * quat
        return Pose(self.pos, quat)

    def flatten(self):
        return self.pos, self.xyzw

    def tree_flatten(self):
        return ((self.pos, self.xyzw), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def flat(self):
        return self.flatten()

    @property
    def shape(self):
        return self.pos.shape[:-1]

    def reshape(self, shape):
        return Pose(self.pos.reshape(shape + (3,)), self.quat.reshape(shape + (4,)))

    def __len__(self):
        return self.pos.shape[0]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        self.current += 1
        if self.current <= len(self):
            return self[self.current - 1]
        raise StopIteration

    def __getitem__(self, index):
        return Pose(self.pos[index], self.quat[index])

    def slice(self, i):
        return Pose(self.pos[i], self.quat[i])

    def as_matrix(self):
        """Return a 4x4 pose matrix."""
        pose_matrix = jnp.zeros((*self.pos.shape[:-1], 4, 4))
        pose_matrix = pose_matrix.at[..., :3, :3].set(
            Rot.from_quat(self.xyzw).as_matrix()
        )
        pose_matrix = pose_matrix.at[..., :3, 3].set(self.pos)
        pose_matrix = pose_matrix.at[..., 3, 3].set(1.0)
        return pose_matrix

    @staticmethod
    def identity():
        """Return the identity transformation."""
        return Pose(jnp.zeros(3), jnp.array([0.0, 0.0, 0.0, 1.0]))

    eye = identity

    def apply(self, vec: Array) -> Array:
        """Apply pose to vectors."""
        return Rot.from_quat(self.xyzw).apply(vec) + self.pos

    def __call__(self, vec: Array) -> Array:
        """Apply pose to vectors."""
        return self.apply(vec)

    def compose(self, pose: "Pose") -> "Pose":
        """Compose with other pose."""
        return Pose(self.apply(pose.pos), multiply_quats(self.xyzw, pose.xyzw))

    def __add__(self, pose: "Pose") -> "Pose":
        return Pose(self.pos + pose.pos, self.quat + pose.quat)

    def __sub__(self, pose: "Pose") -> "Pose":
        return Pose(self.pos - pose.pos, self.quat - pose.quat)

    def scale(self, scale: Float) -> "Pose":
        return Pose(self.pos * scale, self.quat * scale)

    def __mul__(self, scale: Float) -> "Pose":
        return self.scale(scale)

    def __matmul__(self, pose: "Pose") -> "Pose":
        """Compose with other poses."""
        # TODO: Add test, in particular to lock in matmul vs mul.
        return self.compose(pose)

    @staticmethod
    def concatenate_poses(pose_list):
        return Pose(
            jnp.concatenate([pose.pos for pose in pose_list], axis=0),
            jnp.concatenate([pose.quat for pose in pose_list], axis=0),
        )

    @staticmethod
    def stack_poses(pose_list):
        return Pose(
            jnp.stack([pose.pos for pose in pose_list]),
            jnp.stack([pose.quat for pose in pose_list]),
        )

    def split(self, n):
        return [
            Pose(ps, qs)
            for (ps, qs) in zip(
                jnp.array_split(self.pos, n), jnp.array_split(self.quat, n)
            )
        ]

    def inv(self):
        """
        Inverse of pose.

        Note that for rotation matrix R and
        translation vector x we have
        ```
            [[ R x ]  [[ R' -R'x ]  = [[ I  0 ]
             [ 0 1 ]]  [ 0    1  ]]      0  1 ]]
        ```
        where R' is the transpose of R.
        """
        R_inv = Rot.from_quat(self.xyzw).inv()
        return Pose(-R_inv.apply(self.pos), R_inv.as_quat())

    inverse = inv

    def __str__(self):
        return f"Pose(position={repr(self.pos)}, quaternion={repr(self.xyzw)})"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_matrix(matrix):
        """Create an Pose from a 4x4 matrix."""
        return Pose(matrix[..., :3, 3], Rot.from_matrix(matrix[..., :3, :3]).as_quat())

    @staticmethod
    def from_xyzw(xyzw):
        """Create a pose from a quaternion. With zero translation."""
        return Pose(jnp.zeros(3), xyzw)

    from_quat = from_xyzw

    @staticmethod
    def from_pos(position_vec):
        """Create a pose from a vector. With the identity rotation."""
        return Pose(position_vec, jnp.array([0.0, 0.0, 0.0, 1.0]))

    from_translation = from_pos

    @staticmethod
    def from_vec(posxyzw):
        """
        Creates a Pose from a 7-vector [p_x, p_y, p_z, q_x, q_y, q_z, q_w]
        where p is the position and q is the quaternion for the rotation.

        Args:
            posxyzw: Jax array with shape (7,)
        """
        return Pose(posxyzw[:3], posxyzw[3:])

    # TODO: Should we keep that on the Pose class?
    from_position_and_target = camera_from_position_and_target

    sample_uniform_pose = sample_uniform_pose
    sample_gaussian_vmf_pose = sample_gaussian_vmf_pose

    def fit_plane(point_cloud, inlier_threshold, minPoints, maxIteration):
        import pyransac3d

        plane = pyransac3d.Plane()
        plane_eq, _ = plane.fit(
            np.array(point_cloud),
            inlier_threshold,
            minPoints=minPoints,
            maxIteration=maxIteration,
        )
        plane_eq = jnp.array(plane_eq)
        plane_normal = plane_eq[:3]
        point_on_plane = plane_normal * -plane_eq[3]
        plane_x = jnp.cross(plane_normal, np.array([1.0, 0.0, 0.0]))
        plane_y = jnp.cross(plane_normal, plane_x)
        R = jnp.vstack([plane_x, plane_y, plane_normal]).T
        plane_pose = Pose(point_on_plane, Rot.from_matrix(R).as_quat())
        return plane_pose

    def fit_table_plane(
        point_cloud, inlier_threshold, segmentation_threshold, minPoints, maxIteration
    ):
        plane_pose = b3d.Pose.fit_plane(
            point_cloud, inlier_threshold, minPoints, maxIteration
        )
        points_in_plane_frame = plane_pose.inv().apply(point_cloud)
        inliers = jnp.abs(points_in_plane_frame[:, 2]) < inlier_threshold
        inlier_plane_points = points_in_plane_frame[inliers]

        inlier_table_points_seg = b3d.segment_point_cloud(
            inlier_plane_points, segmentation_threshold
        )

        table_points_in_plane_frame = inlier_plane_points[inlier_table_points_seg == 0]

        (cx, cy), (width, height), rotation_deg = cv2.minAreaRect(
            np.array(table_points_in_plane_frame[:, :2])
        )
        pose_shift = b3d.Pose(
            jnp.array([cx, cy, 0.0]),
            b3d.Rot.from_rotvec(
                jnp.array([0.0, 0.0, 1.0]) * jnp.deg2rad(rotation_deg)
            ).as_quat(),
        )
        table_pose = plane_pose @ pose_shift
        table_dims = jnp.array([width, height, 1e-10])
        return table_pose, table_dims
