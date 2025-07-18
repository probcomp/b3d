# ----------------------------------------------------------------------
#  warp_genjax_step.py
#
#  – One source file to import into your project
#  – No manual JAX primitives or batching rules
#  – Uses Warp’s FFI  ➜  jax_callable(…, vmap_method="expand_dims")
#  – Works with *any* batch size you feed through `jax.vmap(trace.update)`
#  – Keeps all heavy geometry/material inside an immutable STATIC dataclass
#  – GenJAX traces only carry body state (q, qd, f)
#
#  ASSUMPTIONS
#    • Per-environment topology is *identical* (20 rigid bodies here)
#     • body_q  layout = 7 floats  (x y z  qx qy qz qw)
#      body_qd layout = 6 floats  (wx wy wz  vx vy vz)
#     • Rigid-contact kernels in this file are *exactly* the ones you posted
#       (broadphase_collision_pairs … integrate_bodies etc.)
#
#  FILL-IN TODOs
#    1. `build_single_env()` – construct **one** environment (20 boxes /
#       “particles”, spheres … whatever) with Warp’s ModelBuilder.
#       Return the builder.
#    2. If your Warp model field names differ, tweak the attribute look-ups
#       inside `model_to_static()` (search # <-- MAP FIELDS).
#
#  HOW TO USE
#     from warp_genjax_step import (
#         build_single_env,          # ➊ write this
#         build_static_and_state,    # ➋ build static + initial state
#         make_warp_step_fn,         # ➌ FFI callback
#     )
#
#     builder_single = build_single_env()
#     static, state0 = build_static_and_state(builder_single)
#
#     warp_step = make_warp_step_fn(static, builder_single)
#
#     # single-env
#     state1 = warp_step(state0, static.sim_dt)
#
#     # 1000 envs in parallel
#     statesB = jax.tree_map(lambda x: jnp.tile(x[None, ...], (1000,) + (1,)*x.ndim), state0)
#     batched_next = jax.vmap(lambda s: warp_step(s, static.sim_dt))(statesB)
#
#  Drop that into GenJAX trace.update() – no extra work.
# ----------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict

import jax
import jax.numpy as jnp
from jax import tree_util
import warp as wp
from warp.jax_experimental.ffi import jax_callable

# ============================================================
# 1.  STATIC & DYNAMIC PYTREES
# ============================================================

@dataclass(frozen=True)
class ModelStatic:
    """Immutable geometry / material / constants – NOT traced by JAX."""
    # counts
    body_count: int
    shape_count: int
    rigid_contact_max: int
    shape_contact_pair_count: int
    shape_ground_contact_pair_count: int
    # scalars
    sim_dt: float
    g: float
    margin: float
    # device arrays (Warp)  ↓↓↓  NOTE: these aren’t traced, live in closure
    shape_contact_pairs: Any
    shape_ground_contact_pairs: Any
    shape_transform: Any
    shape_body: Any
    body_mass: Any
    geo_type: Any
    geo_scale: Any
    geo_source: Any
    geo_thickness: Any
    shape_collision_radius: Any
    ke: Any; kd: Any; kf: Any; ka: Any; mu: Any
    body_com: Any; body_inertia: Any
    body_inv_mass: Any; body_inv_inertia: Any

def _static_flat(x):     return (), x
def _static_unflat(aux, _): return aux
tree_util.register_pytree_node(ModelStatic, _static_flat, _static_unflat)


@dataclass
class WarpState:
    """Time-varying arrays only: (NB,7), (NB,6), (NB,6)"""
    q:  jax.Array
    qd: jax.Array
    f:  jax.Array

def _ws_flat(s): return (s.q, s.qd, s.f), None
def _ws_unflat(aux, ch): return WarpState(*ch)
tree_util.register_pytree_node(WarpState, _ws_flat, _ws_unflat)

# ============================================================
# 2.  SINGLE-ENV  BUILDER  (*** YOU MUST IMPLEMENT ***)
# ============================================================

def build_single_env() -> wp.sim.ModelBuilder:
    """Return a ModelBuilder that creates ONE environment of 20 bodies."""
    builder = wp.sim.ModelBuilder()

    # TODO – replace with your preferred shapes
    for i in range(20):                           # 20 “particles”
        b = builder.add_body(
            origin=wp.transform((i * 0.2, 0.0, 0.0), wp.quat_identity())
        )
        builder.add_shape_sphere(
            body=b,
            pos=wp.vec3(),
            radius=0.05,
            density=1e3,
            ke=1e4, kd=100.0, kf=100.0,
            mu=0.5,
        )

    builder.set_ground_plane(mu=0.5)
    return builder

# ============================================================
# 3.  STATIC  &  INIT-STATE  FROM BUILDER
# ============================================================

def model_to_static(model: wp.sim.Model, *, g=-9.81, dt=1/60, margin=0.02) -> ModelStatic:
    """Extract fields from Warp model (one or batched)."""
    return ModelStatic(
        body_count          = model.body_count,
        shape_count         = model.shape_count,
        rigid_contact_max   = model.rigid_contact_max,
        shape_contact_pair_count   = model.shape_contact_pair_count,
        shape_ground_contact_pair_count = model.shape_ground_contact_pair_count,
        sim_dt = dt,
        g      = g,
        margin = margin,
        # ---- geometry / material  (MAP FIELDS if names differ) ----
        shape_contact_pairs  = model.shape_contact_pairs,
        shape_ground_contact_pairs = model.shape_ground_contact_pairs,
        shape_transform  = model.shape_transform,
        shape_body       = model.shape_body,
        body_mass        = model.body_mass,
        geo_type         = model.shape_geo.type,
        geo_scale        = model.shape_geo.scale,
        geo_source       = model.shape_geo.source,
        geo_thickness    = model.shape_geo.thickness,
        shape_collision_radius = model.shape_collision_radius,
        ke = model.shape_materials.ke,
        kd = model.shape_materials.kd,
        kf = model.shape_materials.kf,
        ka = model.shape_materials.ka,
        mu = model.shape_materials.mu,
        body_com         = model.body_com,
        body_inertia     = model.body_inertia,
        body_inv_mass    = model.body_inv_mass,
        body_inv_inertia = model.body_inv_inertia,
    )

def build_static_and_state(single_builder: wp.sim.ModelBuilder,
                           g=-9.81, dt=1/60, margin=0.02) -> tuple[ModelStatic, WarpState]:
    """Finalize one-env model, get static + initial state (JAX arrays)."""
    model = single_builder.finalize()
    model.ground = True

    state_wp = model.state()
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state_wp)

    static = model_to_static(model, g=g, dt=dt, margin=margin)

    body_q  = jnp.asarray(state_wp.body_q.numpy(), dtype=jnp.float32)
    body_qd = jnp.asarray(state_wp.body_qd.numpy(), dtype=jnp.float32)
    body_f  = jnp.asarray(state_wp.body_f.numpy(), dtype=jnp.float32)

    return static, WarpState(body_q, body_qd, body_f)

# ============================================================
# 4.  CORE  COLLIDE + SIMULATE  (YOUR KERNELS) – imported
# ============================================================

# -- everything you pasted (kernels collide(), simulate() etc.) is assumed
#    to be present in the same module.  We import/use those symbols below.

# ============================================================
# 5.  FFI  WRAPPER  (single call, auto-batched with expand_dims)
# ============================================================

class _StepperFFI:
    """Owns template builder and a cache of batched Warp statics."""

    def __init__(self, single_static: ModelStatic, single_builder: wp.sim.ModelBuilder):
        self.single_static  = single_static
        self.single_builder = single_builder
        self.cache: Dict[int, ModelStatic] = {1: single_static}

    # ---------- helpers ----------
    def _batched_static(self, B: int) -> ModelStatic:
        if B in self.cache:
            return self.cache[B]
        # build batched builder
        top = wp.sim.ModelBuilder()
        for e in range(B):
            xform = wp.transform((e*1.0, 0.0, 0.0), wp.quat_identity())
            top.add_builder(self.single_builder, xform=xform, separate_collision_group=True)
        mB = top.finalize()
        mB.ground = True
        stB = model_to_static(mB, g=self.single_static.g,
                              dt=self.single_static.sim_dt,
                              margin=self.single_static.margin)
        self.cache[B] = stB
        return stB

    # ---------- host-side fused step ----------
    def _ffi_impl(
        self,
        body_q:  wp.array(dtype=wp.transform),        # flattened 1-D
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_f:  wp.array(dtype=wp.spatial_vector),
        dt: float,
        # outputs
        body_q_out:  wp.array(dtype=wp.transform),
        body_qd_out: wp.array(dtype=wp.spatial_vector),
        body_f_out:  wp.array(dtype=wp.spatial_vector),
    ):

        # compute batch size from length
        NB_single = self.single_static.body_count
        total_bodies = body_q.shape[0]
        assert total_bodies % NB_single == 0
        B = total_bodies // NB_single
        st = self._batched_static(B)

        # SCRATCH BUFFERS ------------------------------------------------------------------
        dev   = body_q.device
        RCMax = st.rigid_contact_max
        def iarr(dtype=int):   return wp.zeros(RCMax, dtype=dtype, device=dev)
        def varr():            return wp.zeros(RCMax, dtype=wp.vec3, device=dev)
        rigid_contact_count      = wp.zeros(1, dtype=int, device=dev)
        rigid_contact_broad0     = iarr()
        rigid_contact_broad1     = iarr()
        rigid_contact_point_id   = iarr()
        rigid_contact_shape0     = iarr()
        rigid_contact_shape1     = iarr()
        rigid_contact_point0     = varr()
        rigid_contact_point1     = varr()
        rigid_contact_offset0    = varr()
        rigid_contact_offset1    = varr()
        rigid_contact_normal     = varr()
        rigid_contact_thickness  = wp.zeros(RCMax, dtype=float, device=dev)
        rigid_contact_tids       = iarr()

        # COLLIDE --------------------------------------------------------------------------
        collide(
            st.shape_contact_pair_count,
            st.shape_contact_pairs,
            body_q,
            st.shape_transform,
            st.shape_body,
            st.body_mass,
            st.geo_type,
            st.geo_scale,
            st.geo_source,
            st.geo_thickness,
            st.shape_collision_radius,
            st.rigid_contact_max,
            st.margin,
            st.shape_ground_contact_pair_count,
            st.shape_ground_contact_pairs,
            rigid_contact_count, rigid_contact_broad0, rigid_contact_broad1,
            rigid_contact_point_id,
            rigid_contact_shape0, rigid_contact_shape1,
            rigid_contact_point0, rigid_contact_point1,
            rigid_contact_offset0, rigid_contact_offset1,
            rigid_contact_normal,  rigid_contact_thickness,
            rigid_contact_tids,
            # outputs (same):
            rigid_contact_count, rigid_contact_broad0, rigid_contact_broad1,
            rigid_contact_point_id,
            rigid_contact_shape0, rigid_contact_shape1,
            rigid_contact_point0, rigid_contact_point1,
            rigid_contact_offset0, rigid_contact_offset1,
            rigid_contact_normal, rigid_contact_thickness,
            rigid_contact_tids,
        )

        # SIMULATE -------------------------------------------------------------------------
        simulate(
            st.rigid_contact_max,
            body_q,
            body_qd,
            st.body_com,
            st.ke, st.kd, st.kf, st.ka, st.mu,
            st.geo_thickness,
            st.shape_body,
            rigid_contact_count,
            rigid_contact_point0, rigid_contact_point1,
            rigid_contact_normal,
            rigid_contact_shape0, rigid_contact_shape1,
            st.body_count,
            st.body_inertia, st.body_inv_mass, st.body_inv_inertia,
            st.g,
            dt,
            body_q, body_qd, body_f,
            body_q_out, body_qd_out, body_f_out,
        )

    # ---------- PUBLIC  JAX CALLBACK ----------
    def make_callback(self):
        return jax_callable(
            self._ffi_impl,
            num_outputs=3,
            vmap_method="expand_dims",   # ← fuse vmapped calls
            graph_compatible=True        # CUDA graph capture
        )


# factory that returns a *functional* step(state, dt) -> new_state
def make_warp_step_fn(static: ModelStatic, builder_single):
    """Returns a pure JAX function that steps ONE environment.
       When you vmap it, Warp runs one fused batched step."""

    _cb = _StepperFFI(static, builder_single).make_callback()

    def _step(state: WarpState, dt: float) -> WarpState:
        out_q, out_qd, out_f = _cb(
            state.q, state.qd, state.f, dt,
            output_dims={
                "body_q_out":  state.q.shape,
                "body_qd_out": state.qd.shape,
                "body_f_out":  state.f.shape,
            },
        )
        return WarpState(out_q, out_qd, out_f)

    # jit version for speed
    return jax.jit(_step, static_argnames="dt")


# ============================================================
# 6.  SIMPLE  DEMO  (remove when integrating)
# ============================================================

if __name__ == "__main__":
    wp.verify_cuda_module()          # ensure CUDA available

    builder = build_single_env()     # *** implement bodies here ***
    static, state0 = build_static_and_state(builder)

    # single-env
    step = make_warp_step_fn(static, builder)
    state1 = step(state0, static.sim_dt)
    print("single OK:", state1.q.shape)

    # batched 1000
    B = 1000
    statesB = WarpState(
        jnp.tile(state0.q[None, ...],  (B,1,1)),
        jnp.tile(state0.qd[None, ...], (B,1,1)),
        jnp.tile(state0.f[None, ...],  (B,1,1)),
    )
    vmapped_step = jax.vmap(lambda s: step(s, static.sim_dt))
    statesB2 = vmapped_step(statesB)
    print("batched OK:", statesB2.q.shape)


# -----------------------------------------------------------
# Build once
builder  = build_single_env()
static, state0 = build_static_and_state(builder)
step_fn  = make_warp_step_fn(static, builder)

# GenJAX generative function (single env)
@generative_function
def one_env_generative(s):
    s_next = step_fn(s, static.sim_dt)
    # (yield observations if desired)
    return s_next

# Vectorised across 1000 traces
vmapped_update = jax.vmap(lambda st: one_env_generative.update({}, st))

batch_size = 1000
statesB = WarpState(
    jnp.tile(state0.q[None,...],  (batch_size,1,1)),
    jnp.tile(state0.qd[None,...], (batch_size,1,1)),
    jnp.tile(state0.f[None,...],  (batch_size,1,1)),
)
new_statesB = vmapped_update(statesB)   # ONE Warp launch underneath
