import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Model:
    """This class holds the non-time varying description of the system, i.e.: all geometry, constraints, and parameters used to describe the simulation."""

    def __init__(self, 
                 shape_contact_pair_count, 
                 ground, 
                 shape_ground_contact_pair_count, 
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
                 rigid_contact_max, 
                 rigid_contact_margin, 
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
                 gravity, 
                 ke, 
                 kd, 
                 kf, 
                 ka, 
                 mu,
                 body_count):
        self._shape_contact_pair_count = shape_contact_pair_count
        self._ground = ground
        self._shape_ground_contact_pair_count = shape_ground_contact_pair_count
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
        self._rigid_contact_max = rigid_contact_max
        self._rigid_contact_margin = rigid_contact_margin
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
        self._gravity = gravity
        self._ke = ke
        self._kd = kd
        self._kf = kf
        self._ka = ka
        self._mu = mu
        self._body_count = body_count

    def clear_old_count(self):
        def _clear():
            self._rigid_contact_count = jnp.zeros_like(self._rigid_contact_count)
            self._rigid_contact_broad_shape0 = jnp.full_like(self._rigid_contact_broad_shape0, -1)
            self._rigid_contact_broad_shape1 = jnp.full_like(self._rigid_contact_broad_shape1, -1)

        condition = jnp.logical_or(self._shape_contact_pair_count, jnp.logical_and(self._ground, self._shape_ground_contact_pair_count))
        return jax.lax.cond(condition, lambda _: _clear(), lambda _: None, operand=None)

    def update_attributes(self, **kwargs):
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)

    def tree_flatten(self):
        return ((self._shape_contact_pair_count, self._ground, self._shape_ground_contact_pair_count, self._rigid_contact_count, self._rigid_contact_broad_shape0, self._rigid_contact_broad_shape1, self._shape_contact_pairs, self._shape_transform, self._shape_body, self._body_mass, self._geo_type, self._geo_scale, self._geo_source, self._geo_thickness, self._shape_collision_radius, self._rigid_contact_max, self._rigid_contact_margin, self._rigid_contact_point_id, self._shape_ground_contact_pairs, self._rigid_contact_tids, self._rigid_contact_shape0, self._rigid_contact_shape1, self._rigid_contact_point0, self._rigid_contact_point1, self._rigid_contact_offset0, self._rigid_contact_offset1, self._rigid_contact_normal, self._rigid_contact_thickness, self._body_com, self._body_inertia, self._body_inv_mass, self._body_inv_inertia, self._gravity, self._ke, self._kd, self._kf, self._ka, self._mu, self._body_count), None)

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

    def clear_forces(self):
        self._body_f = jnp.zeros_like(self._body_f)

    def tree_flatten(self):
        return ((self._body_q, self._body_qd, self._body_f), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)