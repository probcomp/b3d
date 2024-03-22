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

import jax_gl_renderer.nvdiffrast.jax as dr


def get_assets_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "assets"
    )

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

@staticmethod
@functools.partial(
    jnp.vectorize,
    signature="(2),(),(m,4,4),()->(3)",
    excluded=(
        4,
        5,
        6,
    ),
)
def interpolate_depth(uv, triangle_id, poses, object_id, vertices, faces, ranges):
    relevant_vertices = vertices[faces[triangle_id-1]]
    pose_of_object = poses[object_id-1]
    relevant_vertices_transformed = relevant_vertices @ pose_of_object.T
    barycentric = jnp.concatenate([uv, jnp.array([1.0 - uv.sum()])])
    interpolated_value = (relevant_vertices_transformed[:,:3] * barycentric.reshape(3,1)).sum(0)
    return interpolated_value

@staticmethod
@functools.partial(
    jnp.vectorize,
    signature="(2),()->(k)",
    excluded=(
        2,
        3,
    ),
)
def interpolate_attribute(uv, triangle_id, faces, attributes):
    relevant_attributes = attributes[faces[triangle_id-1]]
    barycentric = jnp.concatenate([uv, jnp.array([1.0 - uv.sum()])])
    interpolated_value = (relevant_attributes[:,:] * barycentric.reshape(3,1)).sum(0)
    return interpolated_value

class JaxGLRenderer(object):
    def __init__(self, width, height, fx, fy, cx, cy, near, far, num_layers=1024):
        """A renderer for rendering meshes.

        Args:
            intrinsics (bayes3d.camera): The camera intrinsics.
            num_layers (int, optional): The number of scenes to render in parallel. Defaults to 1024.
        """
        self.width, self.height = width, height
        self.resolution = jnp.array([height, width]).astype(jnp.int32)
        self.projection_matrix = projection_matrix_from_intrinsics(width, height, fx, fy, cx, cy, near, far)
        self.renderer_env = dr.RasterizeGLContext(output_db=True)
        self.rasterize = jax.tree_util.Partial(self._rasterize, self)

    # ------------------
    # Rasterization
    # ------------------

    @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _rasterize(self, pose, pos, tri, ranges, projMatrix, resolution):
        return _rasterize_fwd_custom_call(self, pose, pos, tri, ranges, projMatrix, resolution)

    def _rasterize_fwd(self, pose, pos, tri, ranges, projMatrix, resolution):
        rast_out, rast_out_db = _rasterize_fwd_custom_call(self, pose, pos, tri, ranges, projMatrix, resolution)
        saved_tensors = (pose, pos, tri, ranges, projMatrix, resolution, rast_out, rast_out_db)
        return (rast_out, rast_out_db), saved_tensors

    def _rasterize_bwd(self, saved_tensors, diffs):
        pose, pos, tri, ranges, projMatrix, resolution, rast_out, rast_out_db = saved_tensors
        dy, ddb = diffs

        grads = _rasterize_bwd_custom_call(self, pose, pos, tri, ranges, projMatrix, resolution, rast_out, rast_out_db, dy, ddb)
        return grads[0], None, None, None, None, None

    _rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)

    def render_to_barycentrics_many(self, poses, vertices, faces, ranges):
        vertices_h = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
        rast_out, rast_out_aux = self.rasterize(
            poses,
            vertices_h,
            faces,
            ranges,
            self.projection_matrix,
            self.resolution
        )
        uvs = rast_out[...,:2]
        object_ids = rast_out_aux[...,0]
        triangle_ids = rast_out_aux[...,1]
        return uvs, object_ids, triangle_ids

    def render_to_barycentrics(self, pose, vertices, faces, ranges):
        uvs, object_ids, triangle_ids = self.render_to_barycentrics_many(pose[None,...], vertices, faces, ranges)
        return uvs[0], object_ids[0], triangle_ids[0]

    def render_depth_many(self, poses, vertices, faces, ranges):
        vertices_h = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
        rast_out, rast_out_aux = self.rasterize(
            poses,
            vertices_h,
            faces,
            ranges,
            self.projection_matrix,
            self.resolution
        )
        uvs = rast_out[...,:2]
        object_ids = rast_out_aux[...,0]
        triangle_ids = rast_out_aux[...,1]
        mask = object_ids > 0

        interpolated_values = interpolate_depth(uvs, triangle_ids, poses[:,None, None,:,:], object_ids, vertices_h, faces, ranges)
        image = interpolated_values * mask[...,None]
        return image

    def render_depth(self, pose, vertices, faces, ranges):
        return self.render_depth_many(pose[None,...], vertices, faces, ranges)[0]

    def render_attribute_many(self, poses, vertices, faces, ranges, attributes):
        vertices_h = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
        rast_out, rast_out_aux = self.rasterize(
            poses,
            vertices_h,
            faces,
            ranges,
            self.projection_matrix,
            self.resolution
        )
        uvs = rast_out[...,:2]
        object_ids = rast_out_aux[...,0]
        triangle_ids = rast_out_aux[...,1]
        mask = object_ids > 0

        interpolated_values = interpolate_attribute(uvs, triangle_ids, faces, attributes)
        image = interpolated_values * mask[...,None]
        return image
    
    def render_attribute(self, pose, vertices, faces, ranges, attributes):
        return self.render_attribute_many(pose[None,...], vertices, faces, ranges, attributes)[0]


    def render_attribute_faces_many(self, poses, vertices, faces, ranges, attributes):
        vertices_h = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
        rast_out, rast_out_aux = self.rasterize(
            poses,
            vertices_h,
            faces,
            ranges,
            self.projection_matrix,
            self.resolution
        )
        uvs = rast_out[...,:2]
        object_ids = rast_out_aux[...,0]
        triangle_ids = rast_out_aux[...,1]
        mask = object_ids > 0

        interpolated_values = attributes[triangle_ids-1]
        image = interpolated_values * mask[...,None]
        return image
    
    def render_attribute_faces(self, pose, vertices, faces, ranges, attributes):
        return self.render_attribute_faces_many(pose[None,...], vertices, faces, ranges, attributes)[0]

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
def _rasterize_fwd_custom_call(r: "Renderer", pose, pos, tri, ranges, projMatrix, resolution):
    return _build_rasterize_fwd_primitive(r).bind(pose, pos, tri, ranges, projMatrix, resolution)


@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pose, pos, tri, ranges, projection_matrix, resolution):
        if len(pos.shape) != 2 or pos.shape[-1] != 4:
            raise ValueError(
                "Pass in pos aa [num_vertices, 4] sized input"
            )
        if len(pose.shape) != 4:
            raise ValueError(
                "Pos is not 4 dimensional."
            )
        # if len(pose.shape) != 3 or pose.shape[-1] != 4:
        #     raise ValueError(
        #         "Pass in pose aa [num_images, 4, 4] sized input"
        #     )
        num_images = pose.shape[0]

        dtype = dtypes.canonicalize_dtype(pose.dtype)
        int_dtype = dtypes.canonicalize_dtype(np.int32)

        return [
            ShapedArray(
                (num_images, r.height, r.width, 4), dtype
            ),
            ShapedArray(
                (num_images, r.height, r.width, 4), int_dtype
            ),
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx, poses, pos, tri, ranges, projection_matrix, resolution):
        """
        Single-object (one obj represented by tri) rasterization with
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        poses_aval, pos_aval, tri_aval, ranges_aval, projection_matrix_aval, resolution_aval = ctx.avals_in
        # if poses_aval.ndim != 3:
        #     raise NotImplementedError(
        #         f"Only 3D vtx position inputs supported: got {poses_aval.shape}"
        #     )
        # if tri_aval.ndim != 2:
        #     raise NotImplementedError(
        #         f"Only 2D triangle inputs supported: got {tri_aval.shape}"
        #     )
        # if resolution_aval.shape[0] != 2:
        #     raise NotImplementedError(
        #         f"Only 2D resolutions supported: got {resolution_aval.shape}"
        #     )

        np_dtype = np.dtype(poses_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported vtx positions dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangles dtype {tri_aval.dtype}")

        num_images = poses_aval.shape[0]
        num_objects = ranges_aval.shape[0]
        assert num_objects == poses_aval.shape[1], f"Number of poses {poses_aval.shape[1]} should match number of objects {num_objects}"
        num_vertices = pos_aval.shape[0]
        num_triangles = tri_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.height, r.width, 4],
            mlir.dtype_to_ir_type(np_dtype),
        )
        out_shp_dtype_int = mlir.ir.RankedTensorType.get(
            [num_images, r.height, r.width, 4],
            mlir.dtype_to_ir_type(np.dtype(np.int32)),
        )
        opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(
            r.renderer_env.cpp_wrapper, [num_images, num_objects, num_vertices, num_triangles]
        )

        op_name = "jax_rasterize_fwd_gl"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype, out_shp_dtype_int],
            # The inputs:
            operands=[poses, pos, tri, ranges, projection_matrix, resolution],
            backend_config=opaque,
            operand_layouts=[
                (3, 2, 0, 1),
                *default_layouts(
                    pos_aval.shape, tri_aval.shape, ranges_aval.shape, projection_matrix_aval.shape, resolution_aval.shape
                )
            ],
            result_layouts=default_layouts(
                (
                    num_images,
                    r.height,
                    r.width,
                    4,
                ),
                (
                    num_images,
                    r.height,
                    r.width,
                    4,
                ),
            ),
        ).results

    def _render_batch(args, axes):
        pose, pos, tri, ranges, projMatrix, resolution = args
        if pose.ndim != 5:
            raise NotImplementedError("Underlying primitive must operate on 4D poses.")

        original_shape = pose.shape
        poses = jnp.moveaxis(pose, axes[0], 0)
        size_1 = poses.shape[0]
        size_2 = poses.shape[1]
        num_objects = poses.shape[2]
        poses = poses.reshape(size_1 * size_2, num_objects, 4, 4)

        # if poses.shape[1] != indices.shape[0]:
        #     raise ValueError(
        #         f"Poses Original Shape: {original_shape} Poses Shape:  {poses.shape} Indices Shape: {indices.shape}"
        #     )
        # if poses.shape[-2:] != (4, 4):
        #     raise ValueError(
        #         f"Poses Original Shape: {original_shape} Poses Shape:  {poses.shape} Indices Shape: {indices.shape}"
        #     )
        renders, renders2 = _rasterize_fwd_custom_call(r, poses, pos, tri, ranges, projMatrix, resolution)

        renders = renders.reshape(size_1, size_2, *renders.shape[1:])
        renders2 = renders2.reshape(size_1, size_2, *renders2.shape[1:])
        out_axes = 0, 0
        return (renders, renders2), out_axes

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"rasterize_multiple_fwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")
    batching.primitive_batchers[_rasterize_prim] = _render_batch

    return _rasterize_prim


#### BACKWARD ####


# @functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_bwd_custom_call(r: "Renderer", pose, pos, tri, ranges, projection_matrix, resolution, rast_out, rast_out2, dy, ddb):
    return _build_rasterize_bwd_primitive(r).bind(pose, pos, tri, ranges, projection_matrix, resolution, rast_out, rast_out2, dy, ddb)


@functools.lru_cache(maxsize=None)
def _build_rasterize_bwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_bwd_abstract(pose, pos, tri, ranges, projection_matrix, resolution, rast_out, rast_out2, dy, ddb):
        # if len(pos.shape) != 3:
        #     raise ValueError(
        #         "Pass in a [num_images, num_vertices, 4] sized first input"
        #     )
        out_shp = pose.shape
        dtype = dtypes.canonicalize_dtype(pose.dtype)

        return [ShapedArray(out_shp, dtype)]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_bwd_lowering(ctx, pose, pos, tri, ranges, projection_matrix, resolution, rast_out, rast_out2, dy, ddb):
        # Extract the numpy type of the inputs
        (
            poses_aval, pos_aval, tri_aval, ranges_aval,
            projection_matrix_aval, resolution_aval, rast_aval,
            rast_aval2, dy_aval, ddb_aval
        ) = ctx.avals_in

        num_images = poses_aval.shape[0]
        num_objects = ranges_aval.shape[0]
        assert num_objects == poses_aval.shape[1], f"Number of poses {poses_aval.shape[1]} should match number of objects {num_objects}"
        num_vertices = pos_aval.shape[0]
        num_triangles = tri_aval.shape[0]
        depth, height, width = rast_aval.shape[:3]

        print("depth height width", depth, " ", height,  " ", width)

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_bwd_descriptor(
            [num_images, num_objects, num_vertices, num_triangles, height, width]
        )
        
        np_dtype = np.dtype(poses_aval.dtype)
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, num_objects, 4, 4],
            mlir.dtype_to_ir_type(np_dtype),
        )

        op_name = "jax_rasterize_bwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[pose, pos, tri, ranges, projection_matrix, resolution, rast_out, rast_out2, dy, ddb],
            backend_config=opaque,
            operand_layouts=default_layouts(
                poses_aval.shape,
                pos_aval.shape,
                tri_aval.shape,
                ranges_aval.shape,
                projection_matrix_aval.shape,
                resolution_aval.shape,
                rast_aval.shape,
                rast_aval2.shape,
                dy_aval.shape,
                ddb_aval.shape,
            ),
            result_layouts=[(3,2,1,0)],
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"rasterize_multiple_bwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_bwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_bwd_lowering, platform="gpu")

    return _rasterize_prim


# ================================================================================================
# Interpolate
# ================================================================================================

#### FORWARD ####


# # @functools.partial(jax.jit, static_argnums=(0,))
# def _interpolate_fwd_custom_call(
#     r: "Renderer", attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
# ):
#     return _build_interpolate_fwd_primitive(r).bind(
#         attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
#     )


# # @functools.lru_cache(maxsize=None)
# def _build_interpolate_fwd_primitive(r: "Renderer"):
#     _register_custom_calls()
#     # For JIT compilation we need a function to evaluate the shape and dtype of the
#     # outputs of our op for some given inputs

#     def _interpolate_fwd_abstract(
#         attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
#     ):
#         if len(attr.shape) != 3:
#             raise ValueError(
#                 "Pass in a [num_images, num_vertices, num_attributes] sized first input"
#             )
#         num_images, num_vertices, num_attributes = attr.shape
#         _, height, width, _ = rast_out.shape
#         num_tri, _ = tri.shape
#         num_diff_attrs = diff_attrs.shape[0]

#         dtype = dtypes.canonicalize_dtype(attr.dtype)

#         out_abstract = ShapedArray((num_images, height, width, num_attributes), dtype)
#         out_db_abstract = ShapedArray(
#             (num_images, height, width, 2 * num_diff_attrs), dtype
#         )  # empty tensor
#         return [out_abstract, out_db_abstract]

#     # Provide an MLIR "lowering" of the interpolate primitive.
#     def _interpolate_fwd_lowering(
#         ctx, attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
#     ):
#         # Extract the numpy type of the inputs
#         (
#             attr_aval,
#             rast_out_aval,
#             tri_aval,
#             rast_db_aval,
#             _,
#             diff_attr_aval,
#         ) = ctx.avals_in

#         if attr_aval.ndim != 3:
#             raise NotImplementedError(
#                 f"Only 3D attribute inputs supported: got {attr_aval.shape}"
#             )
#         if rast_out_aval.ndim != 4:
#             raise NotImplementedError(
#                 f"Only 4D rast inputs supported: got {rast_out_aval.shape}"
#             )
#         if tri_aval.ndim != 2:
#             raise NotImplementedError(
#                 f"Only 2D triangle tensors supported: got {tri_aval.shape}"
#             )

#         np_dtype = np.dtype(attr_aval.dtype)
#         if np_dtype != np.float32:
#             raise NotImplementedError(f"Unsupported attributes dtype {np_dtype}")
#         if np.dtype(tri_aval.dtype) != np.int32:
#             raise NotImplementedError(f"Unsupported triangle dtype {tri_aval.dtype}")
#         if np.dtype(diff_attr_aval.dtype) != np.int32:
#             raise NotImplementedError(
#                 f"Unsupported diff attribute dtype {diff_attr_aval.dtype}"
#             )

#         num_images, num_vertices, num_attributes = attr_aval.shape
#         depth, height, width = rast_out_aval.shape[:3]
#         num_triangles = tri_aval.shape[0]
#         num_diff_attrs = diff_attr_aval.shape[0]

#         if num_diff_attrs > 0 and rast_db_aval.shape[-1] < num_diff_attrs:
#             raise NotImplementedError(
#                 f"Attempt to propagate bary gradients through {num_diff_attrs} attributes: got {rast_db_aval.shape}"
#             )

#         out_shp_dtype = mlir.ir.RankedTensorType.get(
#             [num_images, height, width, num_attributes], mlir.dtype_to_ir_type(np_dtype)
#         )
#         out_db_shp_dtype = mlir.ir.RankedTensorType.get(
#             [num_images, height, width, 2 * num_diff_attrs],
#             mlir.dtype_to_ir_type(np_dtype),
#         )

#         opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
#             [num_images, num_vertices, num_attributes],
#             [depth, height, width],
#             [num_triangles],
#             num_diff_attrs,  # diff wrt all attributes (TODO)
#         )

#         op_name = "jax_interpolate_fwd"

#         return custom_call(
#             op_name,
#             # Output types
#             result_types=[out_shp_dtype, out_db_shp_dtype],
#             # The inputs:
#             operands=[attr, rast_out, tri, rast_db, diff_attrs],
#             backend_config=opaque,
#             operand_layouts=default_layouts(
#                 attr_aval.shape,
#                 rast_out_aval.shape,
#                 tri_aval.shape,
#                 rast_db_aval.shape,
#                 diff_attr_aval.shape,
#             ),
#             result_layouts=default_layouts(
#                 (
#                     num_images,
#                     height,
#                     width,
#                     num_attributes,
#                 ),
#                 (
#                     num_images,
#                     height,
#                     width,
#                     num_attributes,
#                 ),
#             ),
#         ).results

#     # *********************************************
#     # *  REGISTER THE OP WITH JAX  *
#     # *********************************************
#     _interpolate_prim = core.Primitive(f"interpolate_multiple_fwd_{id(r)}")
#     _interpolate_prim.multiple_results = True
#     _interpolate_prim.def_impl(
#         functools.partial(xla.apply_primitive, _interpolate_prim)
#     )
#     _interpolate_prim.def_abstract_eval(_interpolate_fwd_abstract)

#     # # Connect the XLA translation rules for JIT compilation
#     mlir.register_lowering(_interpolate_prim, _interpolate_fwd_lowering, platform="gpu")

#     return _interpolate_prim


# #### BACKWARD ####


# # @functools.partial(jax.jit, static_argnums=(0,))
# def _interpolate_bwd_custom_call(
#     r: "Renderer",
#     attr,
#     rast_out,
#     tri,
#     dy,
#     rast_db,
#     dda,
#     diff_attrs_all,
#     diff_attrs_list,
# ):
#     return _build_interpolate_bwd_primitive(r).bind(
#         attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
#     )


# # @functools.lru_cache(maxsize=None)
# def _build_interpolate_bwd_primitive(r: "Renderer"):
#     _register_custom_calls()
#     # For JIT compilation we need a function to evaluate the shape and dtype of the
#     # outputs of our op for some given inputs

#     def _interpolate_bwd_abstract(
#         attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
#     ):
#         if len(attr.shape) != 3:
#             raise ValueError(
#                 "Pass in a [num_images, num_vertices, num_attributes] sized first input"
#             )
#         num_images, num_vertices, num_attributes = attr.shape
#         depth, height, width, rast_channels = rast_out.shape
#         depth_db, height_db, width_db, rast_channels_db = rast_db.shape

#         dtype = dtypes.canonicalize_dtype(attr.dtype)

#         g_attr_abstract = ShapedArray((num_images, num_vertices, num_attributes), dtype)
#         g_rast_abstract = ShapedArray((depth, height, width, rast_channels), dtype)
#         g_rast_db_abstract = ShapedArray(
#             (depth_db, height_db, width_db, rast_channels_db), dtype
#         )
#         return [g_attr_abstract, g_rast_abstract, g_rast_db_abstract]

#     # Provide an MLIR "lowering" of the interpolate primitive.
#     def _interpolate_bwd_lowering(
#         ctx, attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
#     ):
#         # Extract the numpy type of the inputs
#         (
#             attr_aval,
#             rast_out_aval,
#             tri_aval,
#             dy_aval,
#             rast_db_aval,
#             dda_aval,
#             _,
#             diff_attr_aval,
#         ) = ctx.avals_in

#         if attr_aval.ndim != 3:
#             raise NotImplementedError(
#                 f"Only 3D attribute inputs supported: got {attr_aval.shape}"
#             )
#         if rast_out_aval.ndim != 4:
#             raise NotImplementedError(
#                 f"Only 4D rast inputs supported: got {rast_out_aval.shape}"
#             )
#         if tri_aval.ndim != 2:
#             raise NotImplementedError(
#                 f"Only 2D triangle tensors supported: got {tri_aval.shape}"
#             )

#         np_dtype = np.dtype(attr_aval.dtype)
#         if np_dtype != np.float32:
#             raise NotImplementedError(f"Unsupported attributes dtype {np_dtype}")
#         if np.dtype(tri_aval.dtype) != np.int32:
#             raise NotImplementedError(f"Unsupported triangle dtype {tri_aval.dtype}")

#         num_images, num_vertices, num_attributes = attr_aval.shape
#         depth, height, width, rast_channels = rast_out_aval.shape
#         depth_db, height_db, width_db, rast_channels_db = rast_db_aval.shape
#         num_triangles = tri_aval.shape[0]
#         num_diff_attrs = diff_attr_aval.shape[0]

#         g_attr_shp_dtype = mlir.ir.RankedTensorType.get(
#             [num_images, num_vertices, num_attributes], mlir.dtype_to_ir_type(np_dtype)
#         )
#         g_rast_shp_dtype = mlir.ir.RankedTensorType.get(
#             [depth, height, width, rast_channels], mlir.dtype_to_ir_type(np_dtype)
#         )
#         g_rast_db_shp_dtype = mlir.ir.RankedTensorType.get(
#             [depth_db, height_db, width_db, rast_channels_db],
#             mlir.dtype_to_ir_type(np_dtype),
#         )

#         opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
#             [num_images, num_vertices, num_attributes],
#             [depth, height, width],
#             [num_triangles],
#             num_diff_attrs,
#         )

#         op_name = "jax_interpolate_bwd"

#         return custom_call(
#             op_name,
#             # Output types
#             result_types=[g_attr_shp_dtype, g_rast_shp_dtype, g_rast_db_shp_dtype],
#             # The inputs:
#             operands=[attr, rast_out, tri, dy, rast_db, dda, diff_attrs_list],
#             backend_config=opaque,
#             operand_layouts=default_layouts(
#                 attr_aval.shape,
#                 rast_out_aval.shape,
#                 tri_aval.shape,
#                 dy_aval.shape,
#                 rast_db_aval.shape,
#                 dda_aval.shape,
#                 diff_attr_aval.shape,
#             ),
#             result_layouts=default_layouts(
#                 (
#                     num_images,
#                     num_vertices,
#                     num_attributes,
#                 ),
#                 (
#                     depth,
#                     height,
#                     width,
#                     rast_channels,
#                 ),
#                 (
#                     depth_db,
#                     height_db,
#                     width_db,
#                     rast_channels_db,
#                 ),
#             ),
#         ).results

#     # *********************************************
#     # *  REGISTER THE OP WITH JAX  *
#     # *********************************************
#     _interpolate_prim = core.Primitive(f"interpolate_multiple_bwd_{id(r)}")
#     _interpolate_prim.multiple_results = True
#     _interpolate_prim.def_impl(
#         functools.partial(xla.apply_primitive, _interpolate_prim)
#     )
#     _interpolate_prim.def_abstract_eval(_interpolate_bwd_abstract)

#     # # Connect the XLA translation rules for JIT compilation
#     mlir.register_lowering(_interpolate_prim, _interpolate_bwd_lowering, platform="gpu")

#     return _interpolate_prim
