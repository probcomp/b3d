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


# Pose Drift Kernels

@Pytree.dataclass
class PhysicsPoseDriftKernel(DriftKernel):
    """A specialized uniform drift kernel with fixed min_val and max_val, with
    additional logics to handle the color channels jointly.

    Support: [max(0.0, prev_value - max_shift), min(1.0, prev_value + max_shift)]
    """

    std: float = Pytree.static()
    concentration: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose, prev_vel, prev_ang_vel):
        keys = jax.random.split(key, 2)
        pos = (
            jax.random.uniform(keys[0], (3,)) * (2 * self.max_shift)
            - self.max_shift
            + prev_pose.position
        )
        quat = jax.random.normal(keys[1], (4,))
        quat = quat / jnp.linalg.norm(quat)
        return Pose(pos, quat)

    def logpdf(self, new_pose, prev_pose) -> ArrayLike:
        position_delta = new_pose.pos - prev_pose.pos
        valid = jnp.all(jnp.abs(position_delta) < self.max_shift)
        position_score = jnp.log(
            (valid * 1.0) * (jnp.ones_like(position_delta) / (2 * self.max_shift))
        ).sum()
        return position_score + jnp.pi**2

@Pytree.dataclass
class GaussianVMFPoseDriftKernel(DriftKernel):
    """A specialized uniform drift kernel with fixed min_val and max_val, with
    additional logics to handle the color channels jointly.

    Support: [max(0.0, prev_value - max_shift), min(1.0, prev_value + max_shift)]
    """

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
