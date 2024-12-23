from abc import abstractmethod

import genjax
from genjax import Pytree
from genjax.typing import ArrayLike, PRNGKey
import jax
import jax.numpy as jnp

from b3d import Pose


@Pytree.dataclass
class DriftKernel(genjax.ExactDensity):
    """An abstract class that defines the common interface for drift kernels."""

    @abstractmethod
    def sample(self, key: PRNGKey, prev_value: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, new_value: ArrayLike, prev_value: ArrayLike) -> ArrayLike:
        raise NotImplementedError


@Pytree.dataclass
class PhysicsKernel(genjax.ExactDensity):
    """An abstract class that defines the common interface for drift kernels."""

    @abstractmethod
    def sample(self, key: PRNGKey, prev_value: ArrayLike, prev_value1: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def logpdf(self, key: PRNGKey, prev_value: ArrayLike, prev_value1: ArrayLike) -> ArrayLike:
        raise NotImplementedError


# Pose Drift Kernels

@Pytree.dataclass
class PhysicsPoseKernel(PhysicsKernel):

    std: float = Pytree.static()
    concentration: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose, prev_vel):
        pos = prev_pose.pos + prev_vel
        # quat = prev_pose.quat + 0.5 * jnp.array([0, prev_ang_vel[0], prev_ang_vel[1], prev_ang_vel[2]]) * prev_pose.quat
        # predict_pose = Pose(pos, quat).normalize()
        predict_pose = Pose(pos, prev_pose.quat)
        return Pose.sample_gaussian_vmf_pose(
            key, predict_pose, self.std, self.concentration
        )

    def logpdf(self, new_pose, prev_pose, prev_vel) -> ArrayLike:
        pos = prev_pose.pos + prev_vel
        # quat = prev_pose.quat + 0.5 * jnp.array([0, prev_ang_vel[0], prev_ang_vel[1], prev_ang_vel[2]]) * prev_pose.quat
        # predict_pose = Pose(pos, quat).normalize()
        predict_pose = Pose(pos, prev_pose.quat)
        return Pose.logpdf_gaussian_vmf_pose(
            new_pose, predict_pose, self.std, self.concentration
        )


@Pytree.dataclass
class GaussianVMFPoseDriftKernel(DriftKernel):

    std: float = Pytree.static()
    concentration: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose):
        return Pose.sample_gaussian_vmf_pose(
            key, prev_pose, self.std, self.concentration
        )

    def logpdf(self, new_pose, prev_pose) -> ArrayLike:
        return Pose.logpdf_gaussian_vmf_pose(
            new_pose, prev_pose, self.std, self.concentration
        )

# Velocities Drift Kernels

@Pytree.dataclass
class GaussianVelocityDriftKernel(DriftKernel):

    std: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_vel):
        return Pose.sample_gaussian_vel(
            key, prev_vel, self.std
        )

    def logpdf(self, new_vel, prev_vel) -> ArrayLike:
        return Pose.logpdf_gaussian_vel(
            new_vel, prev_vel, self.std
        )