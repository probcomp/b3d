# warp_batched_ffi.py  --------------------------------------------------------
from __future__ import annotations
# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jax import tree_util
import warp as wp
from warp.jax_experimental.ffi import jax_callable

# ---------------------------------------------------------------------------
# 1. Static / dynamic containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelStatic:
    """Immutable geometry + material (+ pre-computed contact capacity)."""
    # counts / hyper-params
    body_count:        int
    rigid_contact_max: int
    shape_contact_pair_count:        int
    shape_ground_contact_pair_count: int
    g: float
    dt: float
    margin: float

    # Warp device arrays (these never change at run-time)
    shape_contact_pairs:        Any
    shape_ground_contact_pairs: Any
    shape_transform:            Any
    shape_body:                 Any
    body_mass:                  Any
    geo_type:                   Any
    geo_scale:                  Any
    geo_source:                 Any
    geo_thickness:              Any
    shape_collision_radius:     Any
    body_com:                   Any
    body_inertia:               Any
    body_inv_mass:              Any
    body_inv_inertia:           Any
    ke: Any; kd: Any; kf: Any; ka: Any; mu: Any

# Register as *static* PyTree (no leaves)
tree_util.register_pytree_node(
    ModelStatic,
    lambda m: ((), m),            # children, aux
    lambda aux, _: aux,           # return same object
)


@dataclass
class WarpState:
    """Time-varying body state (layout: N_bodies × {7,6,6})."""
    body_q:  jax.Array  # [N,7]
    body_qd: jax.Array  # [N,6]
    body_f:  jax.Array  # [N,6]
tree_util.register_pytree_node(
    WarpState,
    lambda s: ((s.body_q, s.body_qd, s.body_f), None),
    lambda _, xs: WarpState(*xs),
)

# helpers
zeros_like_f = lambda x: jnp.zeros_like(x)


# ---------------------------------------------------------------------------
# 2. Build single environment (20 “particles” → 20 free boxes here)
# ---------------------------------------------------------------------------
def build_single_env(scale=0.1) -> Tuple[wp.sim.ModelBuilder, ModelStatic]:
    """Return a (template_builder, model_static_single)."""
    builder = wp.sim.ModelBuilder()
    ke = 1.0e4; kd = 100.; kf = 500.
    for i in range(20):                                  # <<< 20 bodies / env
        b = builder.add_body(origin=wp.transform((i*scale*4, 1.0, 0.0),
                                                 wp.quat_identity()))
        builder.add_shape_box(pos=wp.vec3(), hx=scale, hy=scale, hz=scale,
                              body=b, ke=ke, kd=kd, kf=kf,
                              mu=0.3, density=1e3)
    model = builder.finalize()
    model.ground = True

    # pre-fill FK so state has transforms
    state0 = model.state()
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state0)

    # single-env static
    single = ModelStatic(
        body_count            = model.body_count,          # 20
        rigid_contact_max     = model.rigid_contact_max,   # default 64*20
        shape_contact_pair_count        = model.shape_contact_pair_count,
        shape_ground_contact_pair_count = model.shape_ground_contact_pair_count,
        g      = -9.81,
        dt     = 1.0/60.0,
        margin = 0.05,
        shape_contact_pairs        = model.shape_contact_pairs,
        shape_ground_contact_pairs = model.shape_ground_contact_pairs,
        shape_transform            = model.shape_transform,
        shape_body                 = model.shape_body,
        body_mass                  = model.body_mass,
        geo_type                   = model.shape_geo.type,
        geo_scale                  = model.shape_geo.scale,
        geo_source                 = model.shape_geo.source,
        geo_thickness              = model.shape_geo.thickness,
        shape_collision_radius     = model.shape_collision_radius,
        body_com        = model.body_com,
        body_inertia    = model.body_inertia,
        body_inv_mass   = model.body_inv_mass,
        body_inv_inertia= model.body_inv_inertia,
        ke = model.shape_materials.ke,
        kd = model.shape_materials.kd,
        kf = model.shape_materials.kf,
        ka = model.shape_materials.ka,
        mu = model.shape_materials.mu,
    )
    return builder, single, state0


# ---------------------------------------------------------------------------
# 3. Cache of batched (cloned) statics
# ---------------------------------------------------------------------------
_batched_cache: Dict[Tuple[int,int], ModelStatic] = {}  # (id(single), B) -> ModelStatic

def get_batched_static(single: ModelStatic,
                       template_builder: wp.sim.ModelBuilder,
                       B: int) -> ModelStatic:
    if B == 1:
        return single
    key = (id(single), B)
    if key in _batched_cache:
        return _batched_cache[key]

    # replicate env B times
    top = wp.sim.ModelBuilder()
    for e in range(B):
        xform = wp.transform((e*1.0, 0, 0), wp.quat_identity())
        top.add_builder(template_builder, xform=xform,
                        separate_collision_group=True)
    modelB = top.finalize()
    modelB.ground = True

    # make static (copy scalar params from single)
    st = ModelStatic(
        body_count            = modelB.body_count,
        rigid_contact_max     = modelB.rigid_contact_max,
        shape_contact_pair_count        = modelB.shape_contact_pair_count,
        shape_ground_contact_pair_count = modelB.shape_ground_contact_pair_count,
        g      = single.g,
        dt     = single.dt,
        margin = single.margin,
        shape_contact_pairs        = modelB.shape_contact_pairs,
        shape_ground_contact_pairs = modelB.shape_ground_contact_pairs,
        shape_transform            = modelB.shape_transform,
        shape_body                 = modelB.shape_body,
        body_mass                  = modelB.body_mass,
        geo_type                   = modelB.shape_geo.type,
        geo_scale                  = modelB.shape_geo.scale,
        geo_source                 = modelB.shape_geo.source,
        geo_thickness              = modelB.shape_geo.thickness,
        shape_collision_radius     = modelB.shape_collision_radius,
        body_com        = modelB.body_com,
        body_inertia    = modelB.body_inertia,
        body_inv_mass   = modelB.body_inv_mass,
        body_inv_inertia= modelB.body_inv_inertia,
        ke = modelB.shape_materials.ke,
        kd = modelB.shape_materials.kd,
        kf = modelB.shape_materials.kf,
        ka = modelB.shape_materials.ka,
        mu = modelB.shape_materials.mu,
    )
    _batched_cache[key] = st
    return st


# ---------------------------------------------------------------------------
# 4. Foreign-function callback (Warp ↔ JAX)
# ---------------------------------------------------------------------------
def make_warp_step(single_static: ModelStatic,
                   template_builder: wp.sim.ModelBuilder):
    """Return a JAX-callable (body_q, body_qd, body_f, dt) -> updated arrays."""

    # ---- low-level host function (flat arrays) -----------------------------
    def _host_flat(
        body_q:  wp.array(dtype=wp.transform),       # 1-D (N_total,)
        body_qd: wp.array(dtype=wp.spatial_vector), # 1-D
        body_f:  wp.array(dtype=wp.spatial_vector), # 1-D
        dt: float,
        # out buffers
        out_q:  wp.array(dtype=wp.transform),
        out_qd: wp.array(dtype=wp.spatial_vector),
        out_f:  wp.array(dtype=wp.spatial_vector),
    ):
        NB_single = single_static.body_count
        total_bodies = body_q.shape[0]
        assert total_bodies % NB_single == 0
        B = total_bodies // NB_single

        st = get_batched_static(single_static, template_builder, B)

        # --- allocate scratch (one call, could cache on st) -----------------
        device = body_q.device
        rcmax  = st.rigid_contact_max
        iarr = lambda: wp.zeros(rcmax, dtype=int, device=device)
        varr = lambda: wp.zeros(rcmax, dtype=wp.vec3, device=device)
        farr = lambda: wp.zeros(rcmax, dtype=float, device=device)

        rc_count     = wp.zeros(1, dtype=int, device=device)
        rc_broad0    = iarr();  rc_broad1 = iarr();  rc_pid = iarr()
        rc_s0        = iarr();  rc_s1     = iarr()
        rc_p0        = varr();  rc_p1     = varr()
        rc_off0      = varr();  rc_off1   = varr()
        rc_n         = varr();  rc_t      = farr()
        rc_tids      = iarr()

        # -- COLLIDE ---------------------------------------------------------
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
            rc_count, rc_broad0, rc_broad1, rc_pid,
            rc_s0, rc_s1, rc_p0, rc_p1, rc_off0, rc_off1,
            rc_n, rc_t, rc_tids,
            # out (same arrays)
            rc_count, rc_broad0, rc_broad1, rc_pid,
            rc_s0, rc_s1, rc_p0, rc_p1, rc_off0, rc_off1,
            rc_n, rc_t, rc_tids,
        )

        # -- SIMULATE --------------------------------------------------------
        simulate(
            st.rigid_contact_max,
            body_q,
            body_qd,
            st.body_com,
            st.ke, st.kd, st.kf, st.ka, st.mu,
            st.geo_thickness,
            st.shape_body,
            rc_count,
            rc_p0, rc_p1, rc_n,
            rc_s0, rc_s1,
            st.body_count,
            st.body_inertia,
            st.body_inv_mass,
            st.body_inv_inertia,
            st.g, dt,
            body_q, body_qd, body_f,
            out_q, out_qd, out_f,
        )

    # ---- JAX wrapper with expand_dims batching -----------------------------
    step_flat = jax_callable(
        _host_flat, num_outputs=3, vmap_method="sequential"  # *flat* version!
    )

    def warp_step_jax(body_q, body_qd, body_f, dt):
        # body_q has shape  (…?, N, 7)  where leading dims are batch axes
        batch_shape = body_q.shape[:-2]
        flat_len    = int(jnp.prod(jnp.array(batch_shape))) * body_q.shape[-2]

        # let Warp FFI know expected output dims
        out_dims = {"out_q":  flat_len,
                    "out_qd": flat_len,
                    "out_f":  flat_len}

        # Call (FFI automatically flattens last axis when dtype=wp.transform)
        q_new, qd_new, f_new = step_flat(body_q, body_qd, body_f,
                                         dt, output_dims=out_dims)

        # reshape back to original batch dims
        new_shape_q  = batch_shape + body_q.shape[-2:]
        new_shape_vf = batch_shape + body_qd.shape[-2:]
        return (jnp.reshape(q_new,  new_shape_q),
                jnp.reshape(qd_new, new_shape_vf),
                jnp.reshape(f_new,  new_shape_vf))

    # wrap in jax_callable with expand_dims so extra batch axis merges
    warp_step_vmappable = jax_callable(
        warp_step_jax, num_outputs=3, vmap_method="expand_dims"
    )
    return warp_step_vmappable


# ---------------------------------------------------------------------------
# 5. Public helpers GenJAX will call
# ---------------------------------------------------------------------------
def make_single_env():
    """Return (static, state0, step_fn) for one environment."""
    builder, static_single, state0_wp = build_single_env()
    # convert state0_wp to flat JAX arrays
    body_q0  = jnp.asarray(state0_wp.body_q.numpy())   # [20,7]
    body_qd0 = jnp.asarray(state0_wp.body_qd.numpy())  # [20,6]
    body_f0  = jnp.asarray(state0_wp.body_f.numpy())   # [20,6]
    state0   = WarpState(body_q0, body_qd0, body_f0)

    step_fn  = make_warp_step(static_single, builder)
    return static_single, state0, step_fn


# ---------------------------------------------------------------------------
# 6. Example usage (run `python warp_batched_ffi.py` for a demo)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    static, s0, step = make_single_env()

    # --- single env --------------------------------------------------------
    s1 = WarpState(*step(s0.body_q, s0.body_qd, s0.body_f, static.dt))
    print("single-env body_q[0] y-coord:", s1.body_q[0,1])

    # --- 1000 envs via vmap -------------------------------------------------
    B = 1000
    sB0 = WarpState(
        body_q = jnp.tile(s0.body_q[None,...], (B,1,1)),  # (B,20,7)
        body_qd= jnp.tile(s0.body_qd[None,...], (B,1,1)),
        body_f = jnp.zeros_like(jnp.tile(s0.body_qd[None,...], (B,1,1))),
    )

    @jax.jit
    def batch_step(state: WarpState):
        q, qd, f = step(state.body_q, state.body_qd, state.body_f, static.dt)
        return WarpState(q, qd, f)

    batched = jax.vmap(batch_step)(sB0)  # fused Warp run
    print("Batched new shape:", batched.body_q.shape)  # (1000,20,7)


import genjax
from warp_batched_ffi import make_single_env, WarpState

static, state0, warp_step = make_single_env()

@gjax.gen
def physics_gf(state: WarpState):
    next_q, next_qd, next_f = warp_step(state.body_q, state.body_qd,
                                        jnp.zeros_like(state.body_qd),
                                        static.dt)
    next_state = WarpState(next_q, next_qd, next_f)
    return next_state

# 1× trace
trace, weight = physics_gf.importance(jax.random.PRNGKey(0), state0)

# 1000 traces
B = 1000
keys = jax.random.split(jax.random.PRNGKey(42), B)
states = WarpState(jnp.tile(state0.body_q[None,...], (B,1,1)),
                   j
