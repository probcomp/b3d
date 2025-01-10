from abc import abstractmethod

import genjax
from genjax import Pytree
from genjax.typing import ArrayLike, PRNGKey

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
class GaussianVMFPoseDriftKernel(DriftKernel):
    std: float = Pytree.static()
    concentration: float = Pytree.static()

    def sample(self, key: PRNGKey, prev_pose):
        return Pose.sample_gaussian_vmf_pose_approx(
            key, prev_pose, self.std, self.concentration
        )

    def logpdf(self, new_pose, prev_pose) -> ArrayLike:
        return Pose.logpdf_gaussian_vmf_pose_approx(
            new_pose, prev_pose, self.std, self.concentration
        )
