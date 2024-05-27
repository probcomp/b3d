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
import b3d.nvdiffrast.jax as dr
import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from functools import partial

for _name, _value in dr._get_plugin(gl=True).registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

height=100
width=200


renderer_env = dr.RasterizeGLContext(output_db=True)

def _rasterize_fwd_abstract(pose, pos, tri, ranges, projection_matrix, resolution):
    if len(pos.shape) != 2 or pos.shape[-1] != 4:
        raise ValueError("Pass in pos aa [num_vertices, 4] sized input")
    if len(pose.shape) != 4:
        raise ValueError("Pos is not 4 dimensional.")
    num_images = pose.shape[0]

    dtype = dtypes.canonicalize_dtype(pose.dtype)
    int_dtype = dtypes.canonicalize_dtype(np.int32)

    return [
        ShapedArray((num_images, height, width, 4), dtype),
        ShapedArray((num_images, height, width, 4), int_dtype),
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
        [num_images, height, width, 4],
        mlir.dtype_to_ir_type(np_dtype),
    )
    out_shp_dtype_int = mlir.ir.RankedTensorType.get(
        [num_images, height, width, 4],
        mlir.dtype_to_ir_type(np.dtype(np.int32)),
    )
    opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(
        renderer_env.cpp_wrapper,
        [num_images, num_objects, num_vertices, num_triangles, 1024],
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
                height,
                width,
                4,
            ),
            (
                num_images,
                height,
                width,
                4,
            ),
        ),
    ).results

def _rasterize_fwd_vjp(args, tangents):
    poses, pos, tri, ranges, projection_matrix, resolution = args

    output1, output2 = _rasterize_fwd.bind(poses, pos, tri, ranges, projection_matrix, resolution)

    return (
        (output1, output2),
        (jnp.zeros_like(output1), jnp.zeros_like(output2))
    )



_rasterize_fwd = core.Primitive("rasterize_fwd")
_rasterize_fwd.multiple_results = True
_rasterize_fwd.def_impl(partial(xla.apply_primitive, _rasterize_fwd))
_rasterize_fwd.def_abstract_eval(_rasterize_fwd_abstract)

# Connect the XLA translation rules for JIT compilation
# for platform in ["cpu", "gpu"]:
#     mlir.register_lowering(
#         _rasterize_fwd,
#         partial(_rasterize_fwd_lowering, platform=platform),
#         platform=platform)
mlir.register_lowering(_rasterize_fwd, _rasterize_fwd_lowering, platform="gpu")

# Connect the JVP and batching rules
ad.primitive_jvps[_rasterize_fwd] = _rasterize_fwd_jvp
# batching.primitive_batchers[_rasterize_fwd] = _rasterize_fwd_batch




poses, pos, tri =  (
    jnp.zeros((1, 1, 4, 4)),
    jnp.zeros((3, 4)),
    jnp.zeros((1, 3), dtype=jnp.int32),
)

ranges = jnp.array([[0, len(tri)]])
projection_matrix = jnp.eye(4)
resolution = jnp.array([height, width])


output1, output2 = rasterize(poses, pos, tri, ranges, projection_matrix, resolution)

def f(vertices):
    return rasterize(poses, vertices, tri, ranges, projection_matrix, resolution)[0].mean()

jax.grad(f)(pos)
jax.hessian(f)(pos)
jax.jacfwd(f)(pos)
jax.jacrev(f)(pos)



