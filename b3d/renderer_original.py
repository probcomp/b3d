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
import b3d
import b3d.nvdiffrast_jax.nvdiffrast.jax as dr

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



@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def rasterize_prim(self, pos, tri):
    output, = _rasterize_fwd_custom_call(
        self, b3d.pad_with_1(pos) @ self.projection_matrix_t, tri, self.resolution
    )
    return output

def rasterize_fwd(self, pos, tri):
    output, = _rasterize_fwd_custom_call(
        self, b3d.pad_with_1(pos) @ self.projection_matrix_t, tri, self.resolution
    )
    return output, (pos, tri)

def rasterize_bwd(self, saved_tensors, diffs):
    pos, tri = saved_tensors
    return jnp.zeros_like(pos), jnp.zeros_like(tri)

rasterize_prim.defvjp(rasterize_fwd, rasterize_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def interpolate_prim(self, attr, rast, faces):
    output, = _interpolate_fwd_custom_call(
        self, attr, rast, faces
    )
    return output

def interpolate_fwd(self, attr, rast, faces):
    output, = _interpolate_fwd_custom_call(
        self, attr, rast, faces
    )
    return output, (attr, rast, faces)

def interpolate_bwd(self, saved_tensors, diffs):
    attr, rast, faces = saved_tensors
    return jnp.zeros_like(pos), jnp.zeros_like(tri)

interpolate_prim.defvjp(interpolate_fwd, interpolate_bwd)

class RendererOriginal(object):
    def __init__(self, width=200, height=200, fx=150.0, fy=150.0, cx=100.0, cy=100.0, near=0.001, far=10.0):
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
        self.renderer_env = dr.RasterizeGLContext(output_db=False)
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
        self.projection_matrix_t = jnp.transpose(self.projection_matrix)

    def rasterize_many(self, pos, tri):
        return rasterize_prim(self, pos, tri)

    def rasterize(self, pos, tri):
        return self.rasterize_many(pos[None,...], tri)[0]

    def rasterize_original(self, pos, tri):
        return _rasterize_fwd_custom_call(
            self, pos, tri, self.resolution
        )

    def interpolate_many(self, attr, rast, faces):
        return interpolate_prim(self, attr, rast, faces)

    def interpolate(self, attr, rast, faces):
        return self.interpolate_many(attr[None,...], rast[None,...], faces)[0]

    def render_many(self, pos, tri, attr):
        rast = self.rasterize_many(pos, tri)
        return self.interpolate_many(attr, rast, tri)
    
    def render(self, pos, tri, attr):
        return self.render_many(pos[None,...], tri, attr[None,...])[0]

    def render_rgbd_many(self, pos, tri, attr):
        return self.render_many(
            pos, tri,
            jnp.concatenate([attr, pos[...,-1:]],axis=-1)
        )

    def render_rgbd(self, pos, tri, attr):
        return self.render_rgbd_many(
            pos[None,...], tri,
            attr[None,...]
        )[0]

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
def _rasterize_fwd_custom_call(r: "Renderer", pos, tri, resolution):
    return _build_rasterize_fwd_primitive(r).bind(pos, tri, resolution)

@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pos, tri, resolution):
        if len(pos.shape) != 3 or pos.shape[-1] != 4:
            raise ValueError(
                "Pass in a [num_images, num_vertices, 4] sized first input"
            )
        num_images = pos.shape[0]

        dtype = dtypes.canonicalize_dtype(pos.dtype)

        return [
            ShapedArray(
                (num_images, r.height, r.width, 4), dtype
            )
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx, pos, tri, resolution):
        """
        Single-object (one obj represented by tri) rasterization with
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        poses_aval, tri_aval, resolution_aval = ctx.avals_in
        if poses_aval.ndim != 3:
            raise NotImplementedError(
                f"Only 3D vtx position inputs supported: got {poses_aval.shape}"
            )
        if tri_aval.ndim != 2:
            raise NotImplementedError(
                f"Only 2D triangle inputs supported: got {tri_aval.shape}"
            )
        if resolution_aval.shape[0] != 2:
            raise NotImplementedError(
                f"Only 2D resolutions supported: got {resolution_aval.shape}"
            )

        np_dtype = np.dtype(poses_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported vtx positions dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangles dtype {tri_aval.dtype}")

        num_images, num_vertices = poses_aval.shape[:2]
        num_triangles = tri_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.height, r.width, 4],
            mlir.dtype_to_ir_type(np_dtype),
        )

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(
            r.renderer_env.cpp_wrapper, [num_images, num_vertices, num_triangles]
        )

        op_name = "jax_rasterize_fwd_gl"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[pos, tri, resolution],
            backend_config=opaque,
            operand_layouts=default_layouts(
                poses_aval.shape, tri_aval.shape, resolution_aval.shape
            ),
            result_layouts=default_layouts(
                (
                    num_images,
                    r.height,
                    r.width,
                    4,
                ),
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


# @functools.partial(jax.jit, static_argnums=(0,))
def _interpolate_fwd_custom_call(
    r: "Renderer",
    attributes,
    rast,
    faces,
):
    return _build_interpolate_fwd_primitive(r).bind(
        attributes,
        rast,
        faces,
    )

# @functools.lru_cache(maxsize=None)
def _build_interpolate_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _interpolate_fwd_abstract(
        attributes,
        rast,
        faces,
    ):
        _, num_vertices, num_attributes = attributes.shape
        num_images, height, width, _ = rast.shape
        num_tri, _ = faces.shape

        dtype = dtypes.canonicalize_dtype(attributes.dtype)

        out_abstract = ShapedArray((num_images, height, width, num_attributes), dtype)
        return [out_abstract]

    # Provide an MLIR "lowering" of the interpolate primitive.
    def _interpolate_fwd_lowering(
        ctx,
        attributes,
        rast,
        faces,
    ):
        # Extract the numpy type of the inputs
        (attributes_aval, rast_aval, faces_aval) = ctx.avals_in

        _, num_vertices, num_attributes = attributes_aval.shape
        num_images, height, width = rast_aval.shape[:3]
        num_triangles = faces_aval.shape[0]

        np_dtype = np.dtype(rast_aval.dtype)

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, num_attributes], mlir.dtype_to_ir_type(np_dtype)
        )

        opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
            [num_images, num_vertices, num_attributes],
            [num_images, height, width],
            [num_triangles],
            0
        )

        op_name = "jax_interpolate_fwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[attributes, rast, faces],
            backend_config=opaque,
            operand_layouts=default_layouts(
                attributes_aval.shape,
                rast_aval.shape,
                faces_aval.shape,
            ),
            result_layouts=default_layouts(
                (
                    num_images,
                    height,
                    width,
                    num_attributes,
                )
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _interpolate_prim = core.Primitive(f"interpolate_multiple_fwd_{id(r)}")
    _interpolate_prim.multiple_results = True
    _interpolate_prim.def_impl(
        functools.partial(xla.apply_primitive, _interpolate_prim)
    )
    _interpolate_prim.def_abstract_eval(_interpolate_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_interpolate_prim, _interpolate_fwd_lowering, platform="gpu")

    return _interpolate_prim
