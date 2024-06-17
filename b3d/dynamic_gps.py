from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from .types import Array, Int
from .pose import Pose
import jax


@register_pytree_node_class
@dataclass
class DynamicGPS:
    """
    Time dependent Generative Particle System Class.

    Args:
        `positions`: (T,N,3)-array of particle positions in assigned cluster coordinates
        `quaternions`: (T,N,4)-array of particle quaternions (x,y,z,w) in assigned cluster coordinates
        `diagonal_covariances`: (N,3)-array
        `features`: (N,F)-array of particle features
        `cluster_assignments`: (N,)-array of cluster assignments assigning elements to 0-(K-1) clusters
        `cluster_positions`: (T,K,3)-array of cluster positions in world coordiantes
        `cluster_quaternions`: (T,K,4)-array of cluster quaternions in world coorindates
    """

    # Particles defining a distribution of expected features.
    positions: Array
    quaternions: Array
    diagonal_covariances: Array
    features: Array

    # Clustering of particles into clusters.
    cluster_assignments: Array
    cluster_positions: Array
    cluster_quaternions: Array

    @classmethod
    def from_pose_data(
        cls,
        particle_poses,
        particle_covariances,
        features,
        cluster_assignments,
        cluster_poses,
    ):
        return cls(
            positions=particle_poses.pos,
            quaternions=particle_poses.quat,
            diagonal_covariances=particle_covariances,
            features=features,
            cluster_assignments=cluster_assignments,
            cluster_positions=cluster_poses.pos,
            cluster_quaternions=cluster_poses.quat,
        )

    @classmethod
    def from_absolute_pose_data(
        cls,
        absolute_particle_poses,
        particle_covariances,
        features,
        cluster_assignments,
        cluster_poses,
    ):
        relative_particles_poses = cls._compute_relative_poses(
            absolute_particle_poses, cluster_assignments, cluster_poses
        )

        return cls.from_pose_data(
            relative_particles_poses,
            particle_covariances,
            features,
            cluster_assignments,
            cluster_poses,
        )

    # TODO: Initialize from particles and assignments only, putting cluster at center of mass

    def flatten(self):
        return (
            self.positions,
            self.quaternions,
            self.diagonal_covariances,
            self.features,
            self.cluster_assignments,
            self.cluster_positions,
            self.cluster_quaternions,
        )

    @property
    def flat(self):
        return self.flatten()

    def tree_flatten(self):
        return (self.flatten(), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def num_timesteps(self):
        return self.cluster_positions.shape[0]

    @property
    def num_particles(self):
        return self.positions.shape[-2]

    @property
    def feature_dim(self):
        return self.features.shape[-1]

    @property
    def num_clusters(self):
        return self.cluster_positions.shape[1]

    @property
    def poses(self):
        return Pose(self.positions, self.quaternions)

    @poses.setter
    def poses(self, poses):
        self.positions = poses.pos
        self.quaternions = poses.quat

    @property
    def relative_particle_poses(self):
        return self.poses

    @relative_particle_poses.setter
    def relative_particle_poses(self, poses):
        self.positions = poses.pos
        self.quaternions = poses.quat

    @property
    def particle_poses(self):
        return self.poses

    @particle_poses.setter
    def particle_poses(self, poses):
        self.positions = poses.pos
        self.quaternions = poses.quat

    @property
    def cluster_poses(self):
        return Pose(self.cluster_positions, self.cluster_quaternions)

    @cluster_poses.setter
    def cluster_poses(self, poses):
        self.cluster_positions = poses.pos
        self.cluster_quaternions = poses.quat

    def _absolute_particle_poses(self):
        return (
            self.cluster_poses[:, self.cluster_assignments]
            @ self.relative_particle_poses
        )

    @property
    def absolute_particle_poses(self):
        return self._absolute_particle_poses()

    def at_time(self, t: Array | Int):
        if isinstance(t, int):
            ts = jnp.array([t])
        else:
            ts = jnp.array(t)

        return self.__class__(
            positions=self.positions[ts],
            quaternions=self.quaternions[ts],
            diagonal_covariances=self.diagonal_covariances,
            features=self.features,
            cluster_assignments=self.cluster_assignments,
            cluster_positions=self.cluster_positions[ts],
            cluster_quaternions=self.cluster_quaternions[ts],
        )

    @staticmethod
    def _compute_relative_poses(absolute_poses, cluster_assignments, cluster_poses):
        """
        Returns the relative poses with respect to the cluster poses.
        """
        relative_poses = cluster_poses[:, cluster_assignments].inv() @ absolute_poses
        return relative_poses

    def compute_relative_poses(self, absolute_poses):
        """
        Returns the relative poses with respect to the cluster poses.
        """
        return self._compute_relative_poses(
            absolute_poses, self.cluster_assignments, self.cluster_poses
        )

    def change_coordinates(self, pose: Pose):
        """
        Performs a coordinate change to given pose's coordinate system.
        That means we transform the cluster poses to the new coordinate
        system and compute $p^{-1}q$ for each cluster pose $q$.
        """
        cluster_xs, cluster_qs = pose.inv().compose(self.cluster_poses).flat
        return self.__class__(*self.flatten()[:-2], cluster_xs, cluster_qs)

    def _get_particles(self, i):
        if isinstance(i, int):
            i = jnp.array([i])

        return self.__class__(
            positions=self.positions[..., i, :],
            quaternions=self.quaternions[..., i, :],
            diagonal_covariances=self.diagonal_covariances[i],
            features=self.features[i, :],
            cluster_assignments=self.cluster_assignments[i],
            cluster_positions=self.cluster_positions,
            cluster_quaternions=self.cluster_quaternions,
        )

    def get_cluster(self, i: int):
        return self._get_particles(self.cluster_assignments == i)

    # TODO: method that updates the cluster pose without moving the particles in the world.

    def random_color_by_cluster(self, key):
        """Returns an array of colors for each particle based on the cluster assignment."""
        cluster_colors = jax.random.uniform(key, (self.num_clusters, 3))
        return cluster_colors[self.cluster_assignments]
