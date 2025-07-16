from typing import TypeAlias

import genjax
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from tensorflow_probability.substrates import jax as tfp

Array: TypeAlias = jax.Array


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
    def zero_velocity():
        """Return the identity transformation."""
        return Velocity(jnp.zeros(3), jnp.zeros(3))

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
        # angular velocity comes firstin warp by convention
        return f"Velocity(linvel={repr(self.angvel)}, angvel={repr(self.linvel)})"

    def __repr__(self):
        return self.__str__()

    sample_gaussian_vmf_vel_approx = sample_gaussian_vmf_vel_approx
    logpdf_gaussian_vmf_vel_approx = logpdf_gaussian_vmf_vel_approx
