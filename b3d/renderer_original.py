import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
import functools
import os
import b3d.nvdiffrast_original.jax as dr


def projection_matrix_from_intrinsics(w, h, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = jnp.eye(4)
    view = view.at[1:3].set(view[1:3] * -1)

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp = jnp.array(
        [
            [fx, 0.0, -cx, 0.0],
            [0.0, -fy, -cy, 0.0],
            [0.0, 0.0, -near + far, near * far],
            [0.0, 0.0, -1, 0.0],
        ]
    )

    left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
    orth = jnp.array(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ]
    )
    return orth @ persp @ view


class Renderer(object):
    def __init__(self, width, height, fx, fy, cx, cy, near, far, num_layers=2048):
        """
        Triangle mesh renderer.

        Parameters:
            width: int
                Image width.
            height: int
                Image height.
            fx: float
                Focal length x.
            fy: float
                Focal length y.
            cx: float
                Principal point x.
            cy: float
                Principal point y.
            near: float
                Near plane.
            far: float
                Far plane.
            num_layers: int
                Number of layers in the depth buffer.
        """
        self.renderer_env = dr.RasterizeGLContext(output_db=True)
        self.num_layers = num_layers
        self.set_intrinsics(width, height, fx, fy, cx, cy, near, far)

    def set_intrinsics(self, width, height, fx, fy, cx, cy, near, far):
        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.near, self.far = near, far
        self.resolution = jnp.array([height, width]).astype(jnp.int32)
        self.projection_matrix = projection_matrix_from_intrinsics(
            width, height, fx, fy, cx, cy, near, far
        )

    def rasterize(self, pos, tri, ranges, resolution):
        return _rasterize_fwd_custom_call(
            self, pos, tri, ranges, resolution
        )

# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


# Register custom call targets
@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


# ================================================================================================
# Rasterize
# ================================================================================================

#### FORWARD ####

# @functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_fwd_custom_call(
    r: "Renderer", pos, tri, ranges, resolution
):
    return _build_rasterize_fwd_primitive(r).bind(
        pos, tri, ranges, resolution
    )


@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pos, tri, ranges, resolution):
        dtype = dtypes.canonicalize_dtype(pos.dtype)
        int_dtype = dtypes.canonicalize_dtype(np.int32)
        num_images = pos.shape[0]

        return [
            ShapedArray((num_images, r.height, r.width, 4), dtype),
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(
        ctx, pos, tri, ranges, resolution
    ):
        """
        Single-object (one obj represented by tri) rasterization with
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        (
            pos_aval,
            tri_aval,
            ranges_aval,
            resolution_aval,
        ) = ctx.avals_in

        np_dtype = np.dtype(pos_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported vtx positions dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangles dtype {tri_aval.dtype}")

        num_images = pos_aval.shape[0]
        num_objects = ranges_aval.shape[0]
        num_vertices = pos_aval.shape[1]
        num_triangles = tri_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.height, r.width, 4],
            mlir.dtype_to_ir_type(np_dtype),
        )
        opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(
            r.renderer_env.cpp_wrapper,
            [num_images, num_objects, num_vertices, num_triangles, 1],
        )

        op_name = "jax_rasterize_fwd_original_gl"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype,],
            # The inputs:
            operands=[pos, tri, ranges, resolution],
            backend_config=opaque,
            operand_layouts=[
                *default_layouts(
                    pos_aval.shape,
                    tri_aval.shape,
                    ranges_aval.shape,
                    resolution_aval.shape,
                ),
            ],
            result_layouts=default_layouts(
                (
                    num_images,
                    r.height,
                    r.width,
                    4,
                )
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"rasterize_multiple_fwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")

    return _rasterize_prim

