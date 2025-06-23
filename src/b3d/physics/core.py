import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import b3d


@register_pytree_node_class
class Model:
    """This class holds the non-time varying description of the system, i.e.: all geometry, constraints, and parameters used to describe the simulation."""

    def __init__(self, 
                 rigid_contact_count, 
                 rigid_contact_broad_shape0, 
                 rigid_contact_broad_shape1, 
                 shape_contact_pairs, 
                 shape_transform, 
                 shape_body, 
                 body_mass, 
                 geo_type,
                 geo_scale,
                 geo_source,
                 geo_thickness,
                 shape_collision_radius, 
                 rigid_contact_point_id, 
                 shape_ground_contact_pairs, 
                 rigid_contact_tids, 
                 rigid_contact_shape0, 
                 rigid_contact_shape1, 
                 rigid_contact_point0, 
                 rigid_contact_point1, 
                 rigid_contact_offset0, 
                 rigid_contact_offset1, 
                 rigid_contact_normal, 
                 rigid_contact_thickness, 
                 body_com, 
                 body_inertia, 
                 body_inv_mass, 
                 body_inv_inertia, 
                 ke, 
                 kd, 
                 kf, 
                 ka, 
                 mu):
        self._rigid_contact_count = rigid_contact_count
        self._rigid_contact_broad_shape0 = rigid_contact_broad_shape0
        self._rigid_contact_broad_shape1 = rigid_contact_broad_shape1
        self._shape_contact_pairs = shape_contact_pairs
        self._shape_transform = shape_transform
        self._shape_body = shape_body
        self._body_mass = body_mass
        self._geo_type = geo_type
        self._geo_scale = geo_scale
        self._geo_source = geo_source
        self._geo_thickness = geo_thickness
        self._shape_collision_radius = shape_collision_radius
        self._rigid_contact_point_id = rigid_contact_point_id
        self._shape_ground_contact_pairs = shape_ground_contact_pairs
        self._rigid_contact_tids = rigid_contact_tids
        self._rigid_contact_shape0 = rigid_contact_shape0
        self._rigid_contact_shape1 = rigid_contact_shape1
        self._rigid_contact_point0 = rigid_contact_point0
        self._rigid_contact_point1 = rigid_contact_point1
        self._rigid_contact_offset0 = rigid_contact_offset0
        self._rigid_contact_offset1 = rigid_contact_offset1
        self._rigid_contact_normal = rigid_contact_normal
        self._rigid_contact_thickness = rigid_contact_thickness
        self._body_com = body_com
        self._body_inertia = body_inertia
        self._body_inv_mass = body_inv_mass
        self._body_inv_inertia = body_inv_inertia
        self._ke = ke
        self._kd = kd
        self._kf = kf
        self._ka = ka
        self._mu = mu

    def update_attributes(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

    def tree_flatten(self):
        return ((self._rigid_contact_count, self._rigid_contact_broad_shape0, self._rigid_contact_broad_shape1, self._shape_contact_pairs, self._shape_transform, self._shape_body, self._body_mass, self._geo_type, self._geo_scale, self._geo_source, self._geo_thickness, self._shape_collision_radius, self._rigid_contact_point_id, self._shape_ground_contact_pairs, self._rigid_contact_tids, self._rigid_contact_shape0, self._rigid_contact_shape1, self._rigid_contact_point0, self._rigid_contact_point1, self._rigid_contact_offset0, self._rigid_contact_offset1, self._rigid_contact_normal, self._rigid_contact_thickness, self._body_com, self._body_inertia, self._body_inv_mass, self._body_inv_inertia, self._ke, self._kd, self._kf, self._ka, self._mu), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class State:
    """The State object holds all time-varying data for a model."""

    def __init__(self, body_q, body_qd, body_f):
        self._body_q = body_q
        self._body_qd = body_qd
        self._body_f = body_f

    def update_attributes(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

    def from_pos_quat(self):
        _poses = []
        for q in self._body_q:
            pose = jnp.concatenate(q.flat)
            _poses.append(pose)
        self._body_q = jnp.asarray(_poses)

        _vels = []
        for qd in self._body_qd:
            vel = jnp.concatenate(qd.flat)
            _vels.append(vel)
        self._body_qd = jnp.asarray(_vels)

    def to_pos_quat(self):
        _poses = []
        for q in self._body_q:
            pose = b3d.Pose.from_vec(q)
            _poses.append(pose)
        self._body_q = b3d.Pose.stack_poses(_poses)

        _vels = []
        for qd in self._body_qd:
            vel = b3d.Velocity.from_vec(qd)
            _vels.append(vel)
        self._body_qd = b3d.Velocity.stack_velocities(_vels)

    def clear_forces(self):
        self._body_f = jnp.zeros_like(self._body_f)

    def tree_flatten(self):
        return ((self._body_q, self._body_qd, self._body_f), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)