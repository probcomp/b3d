from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

Array: TypeAlias = jax.Array
Float: TypeAlias = Array
Int: TypeAlias = Array
Quaternion: TypeAlias = Array


@register_pytree_node_class
class Model:
    """This class holds the non-time varying description of the system, i.e.: all geometry, constraints, and parameters used to describe the simulation."""

    def __init__(self, shape_contact_pair_count, ground, shape_ground_contact_pair_count, rigid_contact_count, rigid_contact_broad_shape0, rigid_contact_broad_shape1, shape_contact_pairs, shape_transform, shape_body, body_mass, shape_count, shape_geo, shape_collision_radius, rigid_contact_max, rigid_contact_margin, rigid_mesh_contact_max, rigid_contact_point_id, rigid_contact_point_limit, shape_ground_contact_pairs, rigid_contact_tids, rigid_contact_pairwise_counter, rigid_contact_shape0, rigid_contact_shape1, rigid_contact_point0, rigid_contact_point1, rigid_contact_offset0, rigid_contact_offset1, rigid_contact_normal, rigid_contact_thickness, body_com, shape_materials, body_inertia, body_inv_mass, body_inv_inertia, gravity):
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
        self._shape_count = shape_count
        self._shape_geo = shape_geo
        self._shape_collision_radius = shape_collision_radius
        self._rigid_contact_max = rigid_contact_max
        self._rigid_contact_margin = rigid_contact_margin
        self._rigid_mesh_contact_max = rigid_mesh_contact_max
        self._rigid_contact_point_id = rigid_contact_point_id
        self._rigid_contact_point_limit = rigid_contact_point_limit
        self._shape_ground_contact_pairs = shape_ground_contact_pairs
        self._rigid_contact_tids = rigid_contact_tids
        self._rigid_contact_pairwise_counter = rigid_contact_pairwise_counter
        self._rigid_contact_shape0 = rigid_contact_shape0
        self._rigid_contact_shape1 = rigid_contact_shape1
        self._rigid_contact_point0 = rigid_contact_point0
        self._rigid_contact_point1 = rigid_contact_point1
        self._rigid_contact_offset0 = rigid_contact_offset0
        self._rigid_contact_offset1 = rigid_contact_offset1
        self._rigid_contact_normal = rigid_contact_normal
        self._rigid_contact_thickness = rigid_contact_thickness
        self._body_com = body_com
        self._shape_materials = shape_materials
        self._body_inertia = body_inertia
        self._body_inv_mass = body_inv_mass
        self._body_inv_inertia = body_inv_inertia
        self._gravity = gravity

    def tree_flatten(self):
        return ((self._shape_contact_pair_count, self._ground, self._shape_ground_contact_pair_count, self._rigid_contact_count, self._rigid_contact_broad_shape0, self._rigid_contact_broad_shape1, self._shape_contact_pairs, self._shape_transform, self._shape_body, self._body_mass, self._shape_count, self._shape_geo, self._shape_collision_radius, self._rigid_contact_max, self._rigid_contact_margin, self._rigid_mesh_contact_max, self._rigid_contact_point_id, self._rigid_contact_point_limit, self._shape_ground_contact_pairs, self._rigid_contact_tids, self._rigid_contact_pairwise_counter, self._rigid_contact_shape0, self._rigid_contact_shape1, self._rigid_contact_point0, self._rigid_contact_point1, self._rigid_contact_offset0, self._rigid_contact_offset1, self._rigid_contact_normal, self._rigid_contact_thickness, self._body_com, self._shape_materials, self._body_inertia, self._body_inv_mass, self._body_inv_inertia, self._gravity), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class State:
    """The State object holds all time-varying data for a model."""

    def __init__(self, body_q, body_qd, body_f):
        self._body_q = body_q
        self._body_qd = body_qd
        self._body_f = body_f

    def tree_flatten(self):
        return ((self._body_q, self._body_qd, self._body_f), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    

class Physics:
    """The State object holds all time-varying data for a model."""

    def __init__(self, body_q, body_qd, body_f):
        self._body_q = body_q
        self._body_qd = body_qd
        self._body_f = body_f

    def tree_flatten(self):
        return ((self._body_q, self._body_qd, self._body_f), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)