# warp_genjax_step.py
# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Any

import warp as wp
from warp.jax_experimental.ffi import jax_callable

# ------------------------------------------------------------
# 1.  STATIC  vs.  DYNAMIC  containers
# ------------------------------------------------------------

@dataclass(frozen=True)
class ModelStatic:
    """Geometry/material/topology – never traced by JAX."""
    body_count: int
    rigid_contact_max: int
    g: float
    sim_dt: float
    rigid_contact_margin: float

    shape_contact_pair_count: int
    shape_ground_contact_pair_count: int

    # Warp device arrays — references, NOT copied every step
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
    ke: Any
    kd: Any
    kf: Any
    ka: Any
    mu: Any
    body_com: Any
    body_inertia: Any
    body_inv_mass: Any
    body_inv_inertia: Any


# tell JAX “this is static”
tree_util.register_pytree_node(
    ModelStatic,
    lambda ms: ((), ms),            # no children -> static
    lambda aux, _: aux,
)


@dataclass
class WarpState:
    """Time-varying body state (one *environment*)."""
    body_q: jax.Array    # (NB, 7)
    body_qd: jax.Array   # (NB, 6)
    body_f: jax.Array    # (NB, 6)


tree_util.register_pytree_node(
    WarpState,
    lambda s: ((s.body_q, s.body_qd, s.body_f), None),
    lambda _aux, ch: WarpState(*ch),
)

# ------------------------------------------------------------
# 2.  Helpers  –  convert JAX ⇌ Warp
# ------------------------------------------------------------

def jax_to_wp_transform(a: jax.Array) -> wp.array:
    return wp.array(a, dtype=wp.transform, device=wp.get_device())

def jax_to_wp_spatial(a: jax.Array) -> wp.array:
    return wp.array(a, dtype=wp.spatial_vector, device=wp.get_device())

def wp_to_jax(a: wp.array) -> jax.Array:
    return jnp.asarray(a.numpy())

# ------------------------------------------------------------
# 3.  Build template environment  (FILL THIS IN)
# ------------------------------------------------------------

def build_single_env_builder() -> wp.sim.ModelBuilder:
    """Return a ModelBuilder with **exactly one** environment
       (20 bodies, shapes, materials …) – identical to what you
       already do in your init script."""
    builder = wp.sim.ModelBuilder()

    # ------------------------------------------------------------------
    # >>>>  TODO: replicate your own environment construction here  <<<<
    #           (boxes, meshes, spheres … 20 bodies total)
    # ------------------------------------------------------------------

    return builder

# ------------------------------------------------------------
# 4.  Convert Warp model  →  ModelStatic
# ------------------------------------------------------------

def model_to_static(model: wp.sim.Model, *, g=-9.81, dt=1/60.0,
                    margin=0.05) -> ModelStatic:
    """Extract constant arrays / scalars from finalized Warp model."""
    return ModelStatic(
        body_count          = model.body_count,
        rigid_contact_max   = model.rigid_contact_max,
        g                   = g,
        sim_dt              = dt,
        rigid_contact_margin= margin,
        shape_contact_pair_count   = model.shape_contact_pair_count,
        shape_ground_contact_pair_count = model.shape_ground_contact_pair_count,
        shape_contact_pairs = model.shape_contact_pairs,
        shape_ground_contact_pairs = model.shape_ground_contact_pairs,
        shape_transform     = model.shape_transform,
        shape_body          = model.shape_body,
        body_mass           = model.body_mass,
        geo_type            = model.shape_geo.type,
        geo_scale           = model.shape_geo.scale,
        geo_source          = model.shape_geo.source,
        geo_thickness       = model.shape_geo.thickness,
        shape_collision_radius = model.shape_collision_radius,
        ke                  = model.shape_materials.ke,
        kd                  = model.shape_materials.kd,
        kf                  = model.shape_materials.kf,
        ka                  = model.shape_materials.ka,
        mu                  = model.shape_materials.mu,
        body_com            = model.body_com,
        body_inertia        = model.body_inertia,
        body_inv_mass       = model.body_inv_mass,
        body_inv_inertia    = model.body_inv_inertia,
    )

# ------------------------------------------------------------
# 5.  FFI-based batched stepper
# ------------------------------------------------------------

class _StepperCache:
    """Caches a batched ModelStatic (geometry) and the compiled FFI callback
       for each batch size B."""
    def __init__(self, single_builder: wp.sim.ModelBuilder, single_static: ModelStatic):
        self.single_builder = single_builder
        self.single_static  = single_static
        self._batched_static: Dict[int, ModelStatic] = {}
        self._ffi_cb: Dict[int, Any] = {}     # JAX callbacks

    # ---------- batched geometry ----------
    def _get_static(self, B: int) -> ModelStatic:
        if B == 1:
            return self.single_static
        st = self._batched_static.get(B)
        if st is None:
            top = wp.sim.ModelBuilder()
            for e in range(B):
                xform = wp.transform((e*5.0, 0.0, 0.0), wp.quat_identity())
                top.add_builder(
                    self.single_builder,
                    xform=xform,
                    separate_collision_group=True,   # independence!
                )
            modelB = top.finalize()
            modelB.ground = True
            st = model_to_static(modelB, g=self.single_static.g,
                                 dt=self.single_static.sim_dt,
                                 margin=self.single_static.rigid_contact_margin)
            self._batched_static[B] = st
        return st

    # --------------------------------------------------------
    # FFI host *kernel*  (flattened arrays, single Warp call)
    # --------------------------------------------------------
    def _host_kernel(
        self,
        body_q: wp.array(dtype=wp.transform),           # flattened
        body_qd: wp.array(dtype=wp.spatial_vector),
        body_f: wp.array(dtype=wp.spatial_vector),
        dt: float,
        # outputs
        o_q: wp.array(dtype=wp.transform),
        o_qd: wp.array(dtype=wp.spatial_vector),
        o_f: wp.array(dtype=wp.spatial_vector),
    ):
        NB_single = self.single_static.body_count
        total = body_q.shape[0]
        B = total // NB_single
        st = self._get_static(B)

        # ---- allocate scratch (contact) buffers once per call ----
        device = wp.get_device()
        rcmax  = st.rigid_contact_max
        def iarr(): return wp.zeros(rcmax, dtype=int, device=device)
        def farr(): return wp.zeros(rcmax, dtype=float, device=device)
        def varr(): return wp.zeros(rcmax, dtype=wp.vec3, device=device)

        rc_count      = wp.zeros(1, dtype=int, device=device)
        rc_broad0     = iarr(); rc_broad1 = iarr(); rc_pid = iarr()
        rc_shape0     = iarr(); rc_shape1 = iarr()
        rc_p0         = varr(); rc_p1     = varr()
        rc_off0       = varr(); rc_off1   = varr()
        rc_norm       = varr(); rc_thick  = farr(); rc_tids = iarr()

        # ------------------ collide + simulate ------------------
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
            st.rigid_contact_margin,
            st.shape_ground_contact_pair_count,
            st.shape_ground_contact_pairs,
            rc_count,
            rc_broad0,
            rc_broad1,
            rc_pid,
            rc_shape0,
            rc_shape1,
            rc_p0,
            rc_p1,
            rc_off0,
            rc_off1,
            rc_norm,
            rc_thick,
            rc_tids,
            # outputs (same)
            rc_count,
            rc_broad0,
            rc_broad1,
            rc_pid,
            rc_shape0,
            rc_shape1,
            rc_p0,
            rc_p1,
            rc_off0,
            rc_off1,
            rc_norm,
            rc_thick,
            rc_tids,
        )

        simulate(
            st.rigid_contact_max,
            body_q,
            body_qd,
            st.body_com,
            st.ke,
            st.kd,
            st.kf,
            st.ka,
            st.mu,
            st.geo_thickness,
            st.shape_body,
            rc_count,
            rc_p0,
            rc_p1,
            rc_norm,
            rc_shape0,
            rc_shape1,
            st.body_count,
            st.body_inertia,
            st.body_inv_mass,
            st.body_inv_inertia,
            st.g,
            dt,
            body_q,
            body_qd,
            body_f,
            o_q,
            o_qd,
            o_f,
        )

    # -------- compile / cache JAX callback for each B --------
    def get_ffi_callback(self, B: int):
        cb = self._ffi_cb.get(B)
        if cb is None:
            # Wrap host kernel into JAX callable (expand_dims).
            cb = jax_callable(
                self._host_kernel,
                num_outputs=3,
                vmap_method="sequential",   # *this* kernel sees already-flattened arrays
            )
            self._ffi_cb[B] = cb
        return cb


# ------------------------------------------------------------
# 6.  Public step()   (single env, but vmappable)
# ------------------------------------------------------------

def make_warp_step(single_builder: wp.sim.ModelBuilder,
                   single_static: ModelStatic):
    cache = _StepperCache(single_builder, single_static)

    def step(body_q: jax.Array, body_qd: jax.Array,
             body_f: jax.Array, dt: float):
        """
        body_q  : (NB,7)   or (B,NB,7) under vmap
        body_qd : (NB,6)   or (B,NB,6)
        body_f  : same
        """
        NB_single = single_static.body_count
        flat_q  = body_q.reshape(-1, 7)
        flat_qd = body_qd.reshape(-1, 6)
        flat_f  = body_f.reshape(-1, 6)

        B = flat_q.shape[0] // NB_single
        ffi = cache.get_ffi_callback(B)

        out_q, out_qd, out_f = ffi(
            flat_q, flat_qd, flat_f, dt,
            output_dims={
                # give Warp the *flat* dims
                "o_q":  flat_q.shape[0],
                "o_qd": flat_qd.shape[0],
                "o_f":  flat_f.shape[0],
            },
        )

        # reshape back
        return (
            out_q.reshape(body_q.shape),
            out_qd.reshape(body_qd.shape),
            out_f.reshape(body_f.shape),
        )

    # Make JIT-safe version
    step_jit = jax.jit(step)

    # vmappable wrapper that returns WarpState
    def step_state(state: WarpState, dt: float) -> WarpState:
        q, qd, f = step_jit(state.body_q, state.body_qd, state.body_f, dt)
        return WarpState(q, qd, f)

    return step_state


# ------------------------------------------------------------
# 7.  Example usage  (single and batched)
# ------------------------------------------------------------
if __name__ == "__main__":
    wp.init()

    # ---------- build single environment ----------
    template_builder = build_single_env_builder()
    template_model   = template_builder.finalize()
    template_model.ground = True
    static_single    = model_to_static(template_model)

    # ---------- get initial body state ----------
    state0_wp = template_model.state()
    wp.sim.eval_fk(template_model,
                   template_model.joint_q,
                   template_model.joint_qd,
                   None,
                   state0_wp)
    state0 = WarpState(
        body_q  = jnp.asarray(state0_wp.body_q.numpy()),
        body_qd = jnp.asarray(state0_wp.body_qd.numpy()),
        body_f  = jnp.asarray(state0_wp.body_f.numpy()),
    )

    # ---------- make JAX stepper ----------
    step_state = make_warp_step(template_builder, static_single)

    # -------- single env test --------
    s1 = step_state(state0, static_single.sim_dt)
    print("single-env OK", s1.body_q.shape)

    # -------- batched envs (vmapped) --------
    B = 1000
    statesB = WarpState(
        body_q = jnp.tile(state0.body_q[None, ...], (B,1,1)),
        body_qd= jnp.tile(state0.body_qd[None,...], (B,1,1)),
        body_f = jnp.tile(state0.body_f[None, ...], (B,1,1)),
    )

    batched_update = jax.vmap(lambda st: step_state(st, static_single.sim_dt))
    statesB2 = batched_update(statesB)
    print("batched", statesB2.body_q.shape)    # (B, NB, 7)




import genjax
from warp_genjax_step import make_warp_step, WarpState, model_to_static, build_single_env_builder

# build static & stepper once
builder, model_single = build_single_env_builder(), build_single_env_builder().finalize()
static = model_to_static(model_single)
step_state = make_warp_step(builder, static)

@generative_function
def phys_gf(state: WarpState):
    next_state = step_state(state, static.sim_dt)
    #  ... yield observation(s) here ...
    return next_state

# batched traces
B = 1000
states0_batch = WarpState(
    body_q = jnp.tile(state0.body_q[None,...], (B,1,1)),
    body_qd= jnp.tile(state0.body_qd[None,...], (B,1,1)),
    body_f = jnp.tile(state0.body_f[None,...], (B,1,1)),
)

trace  = phys_gf.init(key, states0_batch)
trace2 = trace.update({})          # one physics step – single fused Warp launch
