from typing import TypeAlias

import genjax
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as Rot
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

Array: TypeAlias = jax.Array
Float: TypeAlias = Array
Int: TypeAlias = Array
Quaternion: TypeAlias = Array


# GenJAX Distributions
@jax.jit
def sample_uniform_pose_centered(key, center_pose, low, high):
    keys = jax.random.split(key, 2)
    pos = jax.random.uniform(keys[0], (3,)) * (high - low) + low + center_pose.position
    quat = jax.random.normal(keys[1], (4,))
    quat = quat / jnp.linalg.norm(quat)
    return Pose(pos, quat)


def logpdf_uniform_pose_centered(pose, center_pose, low, high):
    position_delta = pose.pos - center_pose.pos
    valid = (low <= position_delta) & (position_delta <= high)
    position_score = jnp.log(
        (valid * 1.0) * (jnp.ones_like(position_delta) / (high - low))
    )
    return position_score.sum() + jnp.pi**2


uniform_pose_centered = genjax.exact_density(
    sample_uniform_pose_centered, logpdf_uniform_pose_centered
)


def multiply_quats(q1, q2):
    return (Rot.from_quat(q1) * Rot.from_quat(q2)).as_quat()


def multiply_quat_and_vec(q, vs):
    return Rot.from_quat(q).apply(vs)


def choose_good_quat(q):
    """
    If the real part of the quaternion is negative, return the antipodal quaternion,
    which represents the same rotation. If the real part is zero, there is still ambiguity,
    so we'll iteratively resolve that as well.

    Recall that SO(3) is isomorphic to  S^3/x~-x and
    also to D^3/~ where x~-x for x in S^2 = \\partial D^3.
    """
    return jnp.where(
        q[..., [3]] != 0,
        jnp.sign(q[..., [3]]) * q,
        jnp.where(
            q[..., [0]] != 0,
            jnp.sign(q[..., [0]]) * q,
            jnp.where(
                q[..., [1]] != 0, jnp.sign(q[..., [1]]) * q, jnp.sign(q[..., [2]]) * q
            ),
        ),
    )


@jax.jit
def sample_uniform_pose(key, low, high):
    keys = jax.random.split(key, 2)
    pos = jax.random.uniform(keys[0], (3,)) * (high - low) + low
    quat = jax.random.normal(keys[1], (4,))
    quat = quat / jnp.linalg.norm(quat)
    return Pose(pos, quat)


@jax.jit
def sample_uniform_scale(key, low, high):
    key, subkey = jax.random.split(key)
    scale = jax.random.uniform(subkey, (3,)) * (high - low) + low
    return scale


def logpdf_uniform_scale(scale, low, high):
    valid = (low <= scale) & (scale <= high)
    scale_score = jnp.log((valid * 1.0) * (jnp.ones_like(scale) / (high - low)))
    return scale_score.sum() + jnp.pi**2


def logpdf_uniform_pose(pose, low, high):
    position = pose.pos
    valid = (low <= position) & (position <= high)
    position_score = jnp.log((valid * 1.0) * (jnp.ones_like(position) / (high - low)))
    return position_score.sum() + jnp.pi**2


@jax.jit
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
    from b3d.utils import keysplit

    _, keys = keysplit(key, 1, 2)
    var = std**2
    x = jax.random.multivariate_normal(keys[0], mean_pose.pos, var * jnp.eye(3))
    q = tfp.distributions.VonMisesFisher(mean_pose.quat, concentration).sample(
        seed=keys[1]
    )

    return Pose(x, q)


def logpdf_gaussian_vmf_pose(pose, mean_pose, std, concentration):
    translation_score = tfp.distributions.MultivariateNormalDiag(
        mean_pose.pos, jnp.ones(3) * std
    ).log_prob(pose.pos)
    quaternion_score = tfp.distributions.VonMisesFisher(
        mean_pose.quat / jnp.linalg.norm(mean_pose.quat), concentration
    ).log_prob(pose.quat)
    return translation_score + quaternion_score


@jax.jit
def sample_gaussian_vmf_pose_trunc(key, mean_pose, std, concentration):
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
    from b3d.utils import keysplit

    _, keys = keysplit(key, 1, 3)
    var = std**2
    xz = jax.random.multivariate_normal(
        keys[0], jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), var * jnp.eye(2)
    )
    y = genjax.truncated_normal.sample(keys[1], mean_pose.pos[1], std, - 0.00001, jnp.inf)
    q = tfp.distributions.VonMisesFisher(mean_pose.quat, concentration).sample(
        seed=keys[2]
    )

    return Pose(jnp.array([xz[0], y, xz[2]]), q)


def logpdf_gaussian_vmf_pose_trunc(pose, mean_pose, std, concentration):
    translation_score_xz = tfp.distributions.MultivariateNormalDiag(
        jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), jnp.ones(2) * std
    ).log_prob(jnp.array([pose.pos[0], pose.pos[2]]))
    translation_score_y = genjax.truncated_normal.logpdf(
        pose.pos[1], mean_pose.pos[1], std, - 0.00001, jnp.inf
    )
    quaternion_score = tfp.distributions.VonMisesFisher(
        mean_pose.quat / jnp.linalg.norm(mean_pose.quat), concentration
    ).log_prob(pose.quat)
    return translation_score_xz + translation_score_y + quaternion_score


@jax.jit
def sample_gaussian_vmf_pose_approx(key, mean_pose, std, concentration):
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
    from b3d.utils import keysplit

    _, keys = keysplit(key, 1, 3)
    var = std**2
    xz = jax.random.multivariate_normal(
        keys[0], jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), var * jnp.eye(2)
    )
    y = genjax.truncated_normal.sample(keys[1], mean_pose.pos[1], std, - 0.00001, jnp.inf)
    q = jax.random.multivariate_normal(
        keys[1], mean_pose.quat, jnp.eye(4) / concentration
    )

    return Pose(jnp.array([xz[0], y, xz[2]]), q).normalize()


def logpdf_gaussian_vmf_pose_approx(pose, mean_pose, std, concentration):
    translation_score_xz = tfp.distributions.MultivariateNormalDiag(
        jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), jnp.ones(2) * std
    ).log_prob(jnp.array([pose.pos[0], pose.pos[2]]))
    translation_score_y = genjax.truncated_normal.logpdf(
        pose.pos[1], mean_pose.pos[1], std, - 0.00001, jnp.inf
    )
    quaternion_score = tfp.distributions.MultivariateNormalDiag(
        mean_pose.quat, jnp.ones(4) * jnp.sqrt(1 / concentration)
    ).log_prob(pose.quat)
    return translation_score_xz + translation_score_y + quaternion_score


@jax.jit
def sample_gaussian_vmf_pose_xz(key, mean_pose, std, concentration):
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
    from b3d.utils import keysplit

    _, keys = keysplit(key, 1, 2)
    var = std**2
    xz = jax.random.multivariate_normal(
        keys[0], jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), var * jnp.eye(2)
    )
    rot = Rot.from_quat(mean_pose.quat)
    x, y, z = rot.as_euler('xyz')

    y_new = genjax.normal.sample(
        keys[1], y, jnp.sqrt(1 / concentration)
    )
    rot_new = Rot.from_euler('xyz', [x, y_new, z])
    q = rot_new.as_quat()

    return Pose(jnp.array([xz[0], mean_pose.pos[1], xz[2]]), q).normalize()


def logpdf_gaussian_vmf_pose_xz(pose, mean_pose, std, concentration):
    translation_score_xz = tfp.distributions.MultivariateNormalDiag(
        jnp.array([mean_pose.pos[0], mean_pose.pos[2]]), jnp.ones(2) * std
    ).log_prob(jnp.array([pose.pos[0], pose.pos[2]]))

    mean_rot = Rot.from_quat(mean_pose.quat)
    _, mean_y, _ = mean_rot.as_euler('xyz')
    rot = Rot.from_quat(pose.quat)
    _, y, _ = rot.as_euler('xyz')

    quaternion_score = genjax.normal.logpdf(
        y, mean_y, jnp.sqrt(1 / concentration)
    )
    return translation_score_xz + quaternion_score


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
        return self.identity_quaternion

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

    def canonical(self):
        """
        Chooses a canonical representative for the quaternion of the pose, i.e.
        chooses from {q, -q} s.t. q[3] >= 0. Note that if q[3]==0 there is still ambiguity.

        Recall that SO(3) is isomorphic to  S^3/x~-x and
        also to D^3/~ where x~-x for x in S^2 = \\partial D^3.
        """
        quat = self.quat / jnp.linalg.norm(self.quat, axis=-1, keepdims=True)
        return Pose(self.pos, choose_good_quat(quat))

    def flatten(self):
        return self.pos, self.xyzw

    def tree_flatten(self):
        return ((self.pos, self.xyzw), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(self):
        return Pose(jnp.array(self.pos), jnp.array(self.quat))

    @property
    def flat(self):
        return self.flatten()

    @property
    def shape(self):
        return self.pos.shape[:-1]

    def reshape(self, *args):
        shape = jax.tree.leaves(args)
        return Pose(self.pos.reshape(shape + [3]), self.quat.reshape(shape + [4]))

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
    id = identity

    def apply(self, vec: Array) -> Array:
        """Apply pose to vectors."""
        return Rot.from_quat(self.xyzw).apply(vec) + self.pos

    transform_points = apply

    def __call__(self, vec: Array) -> Array:
        """Apply pose to vectors."""
        return self.apply(vec)

    def compose(self, pose: "Pose") -> "Pose":
        """Compose with other pose."""
        return Pose(self.apply(pose.pos), multiply_quats(self.xyzw, pose.xyzw))

    def __add__(self, pose: "Pose") -> "Pose":
        # NOTE: this is useful for gradient updates.
        return Pose(self.pos + pose.pos, self.quat + pose.quat)

    def __sub__(self, pose: "Pose") -> "Pose":
        # NOTE: this is useful for gradient updates.
        return Pose(self.pos - pose.pos, self.quat - pose.quat)

    def scale(self, scale: Float) -> "Pose":
        # NOTE: this is useful for gradient updates.
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

    def concat(self, poses, axis=0):
        return Pose(
            jnp.concatenate([self.pos, poses.pos], axis=axis),
            jnp.concatenate([self.quat, poses.quat], axis=axis),
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
        return Pose(jnp.zeros((*xyzw.shape[:-1], 1)), xyzw)

    from_quat = from_xyzw

    @staticmethod
    def from_pos(position_vec):
        """Create a pose from a vector. With the identity rotation."""
        return Pose(
            position_vec,
            jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (*position_vec.shape[:-1], 1)),
        )

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

    @staticmethod
    def from_pos_matrix(pos, matrix):
        """Create an Pose from a position and a 3x3 matrix."""
        return Pose(pos[..., :3], Rot.from_matrix(matrix[..., :3, :3]).as_quat())

    # TODO: Should we keep that on the Pose class?
    from_position_and_target = camera_from_position_and_target

    sample_uniform_pose = sample_uniform_pose
    sample_uniform_pose_vmap = jax.jit(
        jax.vmap(sample_uniform_pose, in_axes=(0, None, None))
    )
    sample_gaussian_vmf_pose = sample_gaussian_vmf_pose

    sample_gaussian_vmf_pose_jit = jax.jit(sample_gaussian_vmf_pose)
    sample_gaussian_vmf_pose_vmap = jax.vmap(
        sample_gaussian_vmf_pose, in_axes=(0, None, None, None)
    )

    logpdf_gaussian_vmf_pose = logpdf_gaussian_vmf_pose
    logpdf_gaussian_vmf_pose_vmap = jax.jit(
        jax.vmap(logpdf_gaussian_vmf_pose, in_axes=(0, None, None, None))
    )

    sample_gaussian_vmf_pose_approx = sample_gaussian_vmf_pose_approx
    logpdf_gaussian_vmf_pose_approx = logpdf_gaussian_vmf_pose_approx

    sample_gaussian_vmf_pose_trunc = sample_gaussian_vmf_pose_trunc
    logpdf_gaussian_vmf_pose_trunc = logpdf_gaussian_vmf_pose_trunc

    sample_gaussian_vmf_pose_xz = sample_gaussian_vmf_pose_xz
    logpdf_gaussian_vmf_pose_xz = logpdf_gaussian_vmf_pose_xz

    uniform_pose_centered = uniform_pose_centered


@jax.jit
def sample_gaussian_vmf_vel_approx(key, mean_vel, std, concentration):
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
    from b3d.utils import keysplit

    _, keys = keysplit(key, 1, 4)

    lin_vel_dir_sample = jax.random.multivariate_normal(
        keys[0], mean_vel.lin_vel_dir, jnp.eye(3) / concentration
    )
    lin_vel_dir_sample_norm = jnp.linalg.norm(lin_vel_dir_sample)
    lin_vel_dir_new = jnp.where(lin_vel_dir_sample_norm > 0, lin_vel_dir_sample / lin_vel_dir_sample_norm, jnp.zeros_like(lin_vel_dir_sample))
    lin_vel_mag_new = genjax.truncated_normal.sample(keys[1], mean_vel.lin_vel_mag, std, - 0.00001, jnp.inf)

    ang_vel_dir_sample = jax.random.multivariate_normal(
        keys[2], mean_vel.ang_vel_dir, jnp.eye(3) / concentration
    )
    ang_vel_dir_sample_norm = jnp.linalg.norm(ang_vel_dir_sample)
    ang_vel_dir_new = jnp.where(ang_vel_dir_sample_norm > 0, ang_vel_dir_sample / ang_vel_dir_sample_norm, jnp.zeros_like(ang_vel_dir_sample))
    ang_vel_mag_new = genjax.truncated_normal.sample(keys[3], mean_vel.ang_vel_mag, std, - 0.00001, jnp.inf)

    return Velocity(lin_vel_mag_new*lin_vel_dir_new, ang_vel_mag_new*ang_vel_dir_new)

def logpdf_gaussian_vmf_vel_approx(vel, mean_vel, std, concentration):
    linvel_mag_score = genjax.truncated_normal.logpdf(
        vel.lin_vel_mag, mean_vel.lin_vel_mag, std, - 0.00001, jnp.inf
    )
    linvel_dir_score = tfp.distributions.MultivariateNormalDiag(
        mean_vel.lin_vel_dir, jnp.ones(3) * jnp.sqrt(1 / concentration)
    ).log_prob(vel.lin_vel_dir)

    angvel_mag_score = genjax.truncated_normal.logpdf(
        vel.ang_vel_mag, mean_vel.ang_vel_mag, std, - 0.00001, jnp.inf
    )
    angvel_dir_score = tfp.distributions.MultivariateNormalDiag(
        mean_vel.ang_vel_dir, jnp.ones(3) * jnp.sqrt(1 / concentration)
    ).log_prob(vel.ang_vel_dir)

    return linvel_mag_score + linvel_dir_score + angvel_mag_score + angvel_dir_score


@register_pytree_node_class
class Velocity:
    """Velocity class with linear velocity and angular velocity."""

    def __init__(self, lin_vel, ang_vel):
        """
        3D rigid transformation

        position: 3D translation vector
        quaternion: 4D quaternion in xyzw
        """
        self._linvel = lin_vel
        self._angvel = ang_vel

    @property
    def linvel(self):
        return self._linvel

    lin_vel = linvel

    @property
    def angvel(self):
        return self._angvel

    ang_vel = angvel

    @property
    def lin_vel_magnitude(self):
        return jnp.linalg.norm(self._linvel)
    
    @property
    def lin_vel_direction(self):
        norm = jnp.linalg.norm(self._linvel)
        return jnp.where(norm > 0, self._linvel / norm, jnp.zeros_like(self._linvel))
    
    @property
    def ang_vel_magnitude(self):
        return jnp.linalg.norm(self._angvel)
    
    @property
    def ang_vel_direction(self):
        norm = jnp.linalg.norm(self._angvel)
        return jnp.where(norm > 0, self._angvel / norm, jnp.zeros_like(self._angvel))

    lin_vel_mag = lin_vel_magnitude
    lin_vel_dir = lin_vel_direction
    ang_vel_mag = ang_vel_magnitude
    ang_vel_dir = ang_vel_direction

    @staticmethod
    def from_vec(vel):
        """
        Creates a Velocity from a 6-vector 
        """
        return Velocity(vel[:3], vel[3:])
    
    def flatten(self):
        return self.linvel, self.angvel

    def tree_flatten(self):
        return ((self.linvel, self.angvel), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def copy(self):
        return Velocity(jnp.array(self.linvel), jnp.array(self.angvel))

    @property
    def flat(self):
        return self.flatten()

    @property
    def shape(self):
        return self.linvel.shape[:-1]

    def reshape(self, *args):
        shape = jax.tree.leaves(args)
        return Velocity(self.linvel.reshape(shape + [3]), self.angvel.reshape(shape + [3]))

    def __len__(self):
        return self.linvel.shape[0]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        self.current += 1
        if self.current <= len(self):
            return self[self.current - 1]
        raise StopIteration

    def __getitem__(self, index):
        return Velocity(self.linvel[index], self.angvel[index])

    def slice(self, i):
        return Velocity(self.linvel[i], self.angvel[i])

    def __call__(self, vec: Array) -> Array:
        """Apply pose to vectors."""
        return self.apply(vec)

    @staticmethod
    def concatenate_vels(vel_list):
        return Velocity(
            jnp.concatenate([vel.linvel for vel in vel_list], axis=0),
            jnp.concatenate([vel.angvel for vel in vel_list], axis=0),
        )

    def concat(self, vels, axis=0):
        return Velocity(
            jnp.concatenate([self.linvel, vels.linvel], axis=axis),
            jnp.concatenate([self.angvel, vels.angvel], axis=axis),
        )

    @staticmethod
    def stack_velocities(vel_list):
        return Velocity(
            jnp.stack([vel.linvel for vel in vel_list]),
            jnp.stack([vel.angvel for vel in vel_list]),
        )

    def split(self, n):
        return [
            Velocity(ps, qs)
            for (ps, qs) in zip(
                jnp.array_split(self.linvel, n), jnp.array_split(self.angvel, n)
            )
        ]

    def __str__(self):
        return f"Velocity(linvel={repr(self.linvel)}, angvel={repr(self.angvel)})"

    def __repr__(self):
        return self.__str__()

    sample_gaussian_vmf_vel_approx = sample_gaussian_vmf_vel_approx
    logpdf_gaussian_vmf_vel_approx = logpdf_gaussian_vmf_vel_approx
