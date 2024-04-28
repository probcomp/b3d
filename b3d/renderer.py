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
import b3d.nvdiffrast.jax as dr


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
        self._rasterize_partial = jax.tree_util.Partial(self._rasterize, self)

        self.set_intrinsics(width, height, fx, fy, cx, cy, near, far)

    def set_intrinsics(self, width, height, fx, fy, cx, cy, near, far):
        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.resolution = jnp.array([height, width]).astype(jnp.int32)
        self.projection_matrix = projection_matrix_from_intrinsics(
            width, height, fx, fy, cx, cy, near, far
        )

    @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _rasterize(self, pose, pos, tri, ranges, projMatrix, resolution):
        return _rasterize_fwd_custom_call(
            self, pose, pos, tri, ranges, projMatrix, resolution
        )

    def _rasterize_fwd(self, pose, pos, tri, ranges, projMatrix, resolution):
        rast_out, rast_out_db = _rasterize_fwd_custom_call(
            self, pose, pos, tri, ranges, projMatrix, resolution
        )
        saved_tensors = (
            pose,
            pos,
            tri,
            ranges,
            projMatrix,
            resolution,
            rast_out,
            rast_out_db,
        )
        return (rast_out, rast_out_db), saved_tensors

    def _rasterize_bwd(self, saved_tensors, diffs):
        pose, pos, tri, ranges, projMatrix, resolution, rast_out, rast_out_db = (
            saved_tensors
        )
        dy, ddb = diffs

        grads = _rasterize_bwd_custom_call(
            self,
            pose,
            pos,
            tri,
            ranges,
            projMatrix,
            resolution,
            rast_out,
            rast_out_db,
            dy,
            ddb,
        )
        return jnp.zeros_like(pose), None, None, None, None, None

    _rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)

    def interpolate_many(self, attributes, uvs, triangle_ids, faces):
        return _interpolate_fwd_custom_call(self, attributes, uvs, triangle_ids, faces)[
            0
        ]

    def interpolate(self, attributes, uvs, triangle_ids, faces):
        return self.interpolate_many(
            attributes, uvs[None, ...], triangle_ids[None, ...], faces
        )[0]

    def rasterize_many(self, poses, vertices, faces, ranges):
        """
        Rasterize many scenes in parallel. For scene number S and pixel at row i and column j,
        uvs[S, i, j] contains the u and v barycentric coordinates of the intersection point with
        triangle index at triangle_ids[S, i, j] - 1 which is on object index object_ids[S, i, j] - 1.
        And the z coordinate of the intersection point is z[S, i, j].
        If the pixel's ray did not intersect any triangle, the values in corresponding pixel is 0.

        Parameters:
            poses: float array, shape (num_scenes, num_objects, 4, 4)
                Object pose matrix.
            vertices: float array, shape (num_vertices, 3)
                Vertex position matrix.
            faces: int array, shape (num_triangles, 3)
                Faces Triangle matrix. The integers correspond to rows in the vertices matrix.
            ranges: int array, shape (num_objects, 2)
                Ranges matrix with the 2 elements specify start indices and counts into faces.
        Outputs:
            uvs: float array, shape (num_scenes, height, width, 2)
                UV coordinates of the intersection point on the triangle. Zeros if the pixel ray doesn't collide a triangle.
            object_ids: int array, shape (num_scenes, height, width)
                Index + 1 of the object that this pixel ray intersections. Zero if the pixel ray doesn't collide a triangle.
            triangle_ids: int array, shape (num_scenes, height, width)
                Index + 1 of the triangle face that this pixel ray intersections. Zero if the pixel ray doesn't collide a triangle.
            zs: float array, shape (num_scenes, height, width)
                Depth of the intersection point. Zero if the pixel ray doesn't collide a triangle.
        """
        vertices_h = jnp.concatenate(
            [vertices, jnp.ones((vertices.shape[0], 1))], axis=-1
        )
        rast_out, rast_out_aux = self._rasterize_partial(
            poses,
            vertices_h,
            faces,
            ranges,
            self.projection_matrix,
            self.resolution,
        )
        uvs = rast_out[..., :2]
        zs = rast_out[..., 3]
        object_ids = rast_out_aux[..., 0]
        triangle_ids = rast_out_aux[..., 1]
        return uvs, object_ids, triangle_ids, zs

    def rasterize(self, pose, vertices, faces, ranges):
        """
        Rasterize a singe scene.

        Parameters:
            poses: float array, shape (num_objects, 4, 4)
                Object pose matrix.
            vertices: float array, shape (num_vertices, 3)
                Vertex position matrix.
            faces: int array, shape (num_triangles, 3)
                Faces Triangle matrix. The integers correspond to rows in the vertices matrix.
            ranges: int array, shape (num_objects, 2)
                Ranges matrix with the 2 elements specify start indices and counts into faces.
        Outputs:
            uvs: float array, shape (height, width, 2)
                UV coordinates of the intersection point on the triangle. Zeros if the pixel ray doesn't collide a triangle.
            object_ids: int array, shape (height, width)
                Index + 1 of the object that this pixel ray intersections. Zero if the pixel ray doesn't collide a triangle.
            triangle_ids: int array, shape (height, width)
                Index + 1 of the triangle face that this pixel ray intersections. Zero if the pixel ray doesn't collide a triangle.
            zs: float array, shape (height, width)
                Depth of the intersection point. Zero if the pixel ray doesn't collide a triangle.
        """
        uvs, object_ids, triangle_ids, zs = self.rasterize_many(
            pose[None, ...], vertices, faces, ranges
        )
        return uvs[0], object_ids[0], triangle_ids[0], zs[0]

    def render_attribute_many(self, poses, vertices, faces, ranges, attributes):
        """
        Render many scenes to an image by rasterizing and then interpolating attributes.

        Parameters:
            poses: float array, shape (num_scenes, num_objects, 4, 4)
                Object pose matrix.
            vertices: float array, shape (num_vertices, 3)
                Vertex position matrix.
            faces: int array, shape (num_triangles, 3)
                Faces Triangle matrix. The integers correspond to rows in the vertices matrix.
            ranges: int array, shape (num_objects, 2)
                Ranges matrix with the 2 elements specify start indices and counts into faces.
            attributes: float array, shape (num_vertices, num_attributes)
                Attributes corresponding to the vertices

        Outputs:
            image: float array, shape (num_scenes, height, width, num_attributes)
                At each pixel the value is the barycentric interpolation of the attributes corresponding to the
                3 vertices of the triangle with which the pixel's ray intersected. If the pixel's ray does not intersect
                any triangle the value at that pixel will be 0s.
            zs: float array, shape (num_scenes, height, width)
                Depth of the intersection point. Zero if the pixel ray doesn't collide a triangle.
        """
        uvs, object_ids, triangle_ids, zs = self.rasterize_many(
            poses, vertices, faces, ranges
        )
        mask = object_ids > 0

        interpolated_values = self.interpolate_many(
            attributes, uvs, triangle_ids, faces
        )
        image = interpolated_values * mask[..., None]
        #  + (1 - mask[...,None]) * 1.0
        return image, zs

    def render_attribute(self, pose, vertices, faces, ranges, attributes):
        """
        Render a single scenes to an image by rasterizing and then interpolating attributes.

        Parameters:
            poses: float array, shape (num_objects, 4, 4)
                Object pose matrix.
            vertices: float array, shape (num_vertices, 3)
                Vertex position matrix.
            faces: int array, shape (num_triangles, 3)
                Faces Triangle matrix. The integers correspond to rows in the vertices matrix.
            ranges: int array, shape (num_objects, 2)
                Ranges matrix with the 2 elements specify start indices and counts into faces.
            attributes: float array, shape (num_vertices, num_attributes)
                Attributes corresponding to the vertices

        Outputs:
            image: float array, shape (height, width, num_attributes)
                At each pixel the value is the barycentric interpolation of the attributes corresponding to the
                3 vertices of the triangle with which the pixel's ray intersected. If the pixel's ray does not intersect
                any triangle the value at that pixel will be 0s.
            zs: float array, shape (height, width)
                Depth of the intersection point. Zero if the pixel ray doesn't collide a triangle.
        """
        image, zs = self.render_attribute_many(
            pose[None, ...], vertices, faces, ranges, attributes
        )
        return image[0], zs[0]


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
    r: "Renderer", pose, pos, tri, ranges, projMatrix, resolution
):
    return _build_rasterize_fwd_primitive(r).bind(
        pose, pos, tri, ranges, projMatrix, resolution
    )


@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pose, pos, tri, ranges, projection_matrix, resolution):
        if len(pos.shape) != 2 or pos.shape[-1] != 4:
            raise ValueError("Pass in pos aa [num_vertices, 4] sized input")
        if len(pose.shape) != 4:
            raise ValueError("Pos is not 4 dimensional.")
        # if len(pose.shape) != 3 or pose.shape[-1] != 4:
        #     raise ValueError(
        #         "Pass in pose aa [num_images, 4, 4] sized input"
        #     )
        num_images = pose.shape[0]

        dtype = dtypes.canonicalize_dtype(pose.dtype)
        int_dtype = dtypes.canonicalize_dtype(np.int32)

        return [
            ShapedArray((num_images, r.height, r.width, 4), dtype),
            ShapedArray((num_images, r.height, r.width, 4), int_dtype),
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(
        ctx, poses, pos, tri, ranges, projection_matrix, resolution
    ):
        """
        Single-object (one obj represented by tri) rasterization with
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        (
            poses_aval,
            pos_aval,
            tri_aval,
            ranges_aval,
            projection_matrix_aval,
            resolution_aval,
        ) = ctx.avals_in
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
        assert (
            num_objects == poses_aval.shape[1]
        ), f"Number of poses {poses_aval.shape[1]} should match number of objects {num_objects}"
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
            r.renderer_env.cpp_wrapper,
            [num_images, num_objects, num_vertices, num_triangles, r.num_layers],
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
                    pos_aval.shape,
                    tri_aval.shape,
                    ranges_aval.shape,
                    projection_matrix_aval.shape,
                    resolution_aval.shape,
                ),
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
        renders, renders2 = _rasterize_fwd_custom_call(
            r, poses, pos, tri, ranges, projMatrix, resolution
        )

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
def _rasterize_bwd_custom_call(
    r: "Renderer",
    pose,
    pos,
    tri,
    ranges,
    projection_matrix,
    resolution,
    rast_out,
    rast_out2,
    dy,
    ddb,
):
    return _build_rasterize_bwd_primitive(r).bind(
        pose,
        pos,
        tri,
        ranges,
        projection_matrix,
        resolution,
        rast_out,
        rast_out2,
        dy,
        ddb,
    )


@functools.lru_cache(maxsize=None)
def _build_rasterize_bwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_bwd_abstract(
        pose,
        pos,
        tri,
        ranges,
        projection_matrix,
        resolution,
        rast_out,
        rast_out2,
        dy,
        ddb,
    ):
        # if len(pos.shape) != 3:
        #     raise ValueError(
        #         "Pass in a [num_images, num_vertices, 4] sized first input"
        #     )
        out_shp = pose.shape
        dtype = dtypes.canonicalize_dtype(pose.dtype)

        return [ShapedArray(out_shp, dtype)]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_bwd_lowering(
        ctx,
        pose,
        pos,
        tri,
        ranges,
        projection_matrix,
        resolution,
        rast_out,
        rast_out2,
        dy,
        ddb,
    ):
        # Extract the numpy type of the inputs
        (
            poses_aval,
            pos_aval,
            tri_aval,
            ranges_aval,
            projection_matrix_aval,
            resolution_aval,
            rast_aval,
            rast_aval2,
            dy_aval,
            ddb_aval,
        ) = ctx.avals_in

        num_images = poses_aval.shape[0]
        num_objects = ranges_aval.shape[0]
        assert (
            num_objects == poses_aval.shape[1]
        ), f"Number of poses {poses_aval.shape[1]} should match number of objects {num_objects}"
        num_vertices = pos_aval.shape[0]
        num_triangles = tri_aval.shape[0]
        depth, height, width = rast_aval.shape[:3]

        print("depth height width", depth, " ", height, " ", width)

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
            operands=[
                pose,
                pos,
                tri,
                ranges,
                projection_matrix,
                resolution,
                rast_out,
                rast_out2,
                dy,
                ddb,
            ],
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
            result_layouts=[(3, 2, 1, 0)],
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


# @functools.partial(jax.jit, static_argnums=(0,))
def _interpolate_fwd_custom_call(
    r: "Renderer",
    attributes,
    uvs,
    triangle_ids,
    faces,
):
    return _build_interpolate_fwd_primitive(r).bind(
        attributes, uvs, triangle_ids, faces
    )


# @functools.lru_cache(maxsize=None)
def _build_interpolate_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _interpolate_fwd_abstract(
        attributes,
        uvs,
        triangle_ids,
        faces,
    ):
        num_vertices, num_attributes = attributes.shape
        num_images, height, width, _ = uvs.shape
        num_tri, _ = faces.shape

        dtype = dtypes.canonicalize_dtype(attributes.dtype)

        out_abstract = ShapedArray((num_images, height, width, num_attributes), dtype)
        return [out_abstract]

    # Provide an MLIR "lowering" of the interpolate primitive.
    def _interpolate_fwd_lowering(
        ctx,
        attributes,
        uvs,
        triangle_ids,
        faces,
    ):
        # Extract the numpy type of the inputs
        (attributes_aval, uvs_aval, triangle_ids_aval, faces_aval) = ctx.avals_in

        num_vertices, num_attributes = attributes_aval.shape
        num_images, height, width = uvs_aval.shape[:3]
        num_triangles = faces_aval.shape[0]

        np_dtype = np.dtype(uvs_aval.dtype)

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, num_attributes], mlir.dtype_to_ir_type(np_dtype)
        )

        opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
            [num_images, num_vertices, num_attributes],
            [num_images, height, width],
            [num_triangles],
        )

        op_name = "jax_interpolate_fwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[attributes, uvs, triangle_ids, faces],
            backend_config=opaque,
            operand_layouts=default_layouts(
                attributes_aval.shape,
                uvs_aval.shape,
                triangle_ids_aval.shape,
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

    def _render_batch_interp(args, axes):
        attributes, uvs, triangle_ids, faces = args

        original_shape_uvs = uvs.shape
        original_shape_triangle_ids = triangle_ids.shape

        uvs = jnp.moveaxis(uvs, axes[1], 0)
        size_1 = uvs.shape[0]
        size_2 = uvs.shape[1]
        triangle_ids = jnp.moveaxis(triangle_ids, axes[2], 0)

        uvs = uvs.reshape(uvs.shape[0] * uvs.shape[1], *uvs.shape[2:])
        triangle_ids = triangle_ids.reshape(
            triangle_ids.shape[0] * triangle_ids.shape[1], *triangle_ids.shape[2:]
        )

        image = _interpolate_fwd_custom_call(r, attributes, uvs, triangle_ids, faces)[0]

        image = image.reshape(size_1, size_2, *image.shape[1:])
        out_axes = (0,)
        return (image,), out_axes

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
    batching.primitive_batchers[_interpolate_prim] = _render_batch_interp

    return _interpolate_prim


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
