import jax.numpy as jnp
import jax
import b3d
from b3d import Pose

BIGNUM = 1e6

#
# Note: right now, in RGBD rendering mode,
# the rendered depth values are get truncated
# to the min/max depth values on the triangles being hit.
# This is because we do a barycentric interpolation for depth.
# But I could instead return the z value of the point where the pixel's ray
# intersects the triangle plane, which may be farther or closer in depth
# than the triangle's min/max depth
# (which is truncated at the vertices of the triangle.)
#

class DifferentiableRendererHyperparams:
    """
    Hyperparameters for the differentiable renderer.
    - WINDOW: specifies the size of the window around each pixel.
        The window will have size (2 * WINDOW + 1) x (2 * WINDOW + 1).
    - SIGMA, GAMMA, EPSILON: Hyperparameters for softras-style scoring.
        See https://arxiv.org/pdf/1904.01786 for details.
        Decreasing sigma will increase the sharpness of the boundaries.
        Decreasing gamma will increase the effect of the z value on a triangle's
        contribution to the pixel.
        Increasing epsilon will increase the weight of the background (ie. of
        contributions not corresponding to any triangle).
        See b3d.differentiable_renderer.DEFAULT_HYPERPARAMS for one suggested set of values.
    """
    def __init__(self, window, sigma, gamma, epsilon):
        self.WINDOW = window
        self.SIGMA = sigma
        self.GAMMA = gamma
        self.EPSILON = epsilon

class HyperparamsAndIntrinsics:
    """
    Internal data structure for passing around hyperparameters and camera intrinsics
    within the differentiable renderer.
    """
    def __init__(self, hyperparams, fx, fy, cx, cy):
        self.hyperparams = hyperparams
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

DEFAULT_HYPERPARAMS = DifferentiableRendererHyperparams(3, 5e-5, 0.25, -1)

#####################################
### Top-level rendering functions ###
#####################################

def render_to_rgbd_dist_params(
    renderer: b3d.Renderer,
    vertices,
    faces,
    vertex_rgbs,
    hyperparams=DEFAULT_HYPERPARAMS
):
    """
    Variant of `render_to_dist_params` for rendering RGBD.
    Accepts RGB vertex attributes and appends depth to them.
    """
    vertex_depths = vertices[:, 2]
    vertex_rgbds = jnp.concatenate([vertex_rgbs, vertex_depths[:, None]], axis=1)
    return render_to_dist_params(renderer, vertices, faces, vertex_rgbds, hyperparams)

def render_to_dist_params(renderer, vertices, faces, vertex_attributes, hyperparams=DEFAULT_HYPERPARAMS, transform=Pose.identity()):
    """
    Differentiable rendering to parameters for a per-pixel
    categorical distribution over attributes (e.g. RGB or RGBD).

    Args:
    - renderer: b3d.Renderer
    - vertices: (V, 3)
    - faces: (F, 3)
    - vertex_attributes: (F, A) [A=3 for RGB; A=4 for RGBD]
    - hyperparams: DifferentiableRendererHyperparams
    Returns:
    - weights (H, W, U)
    - attributes (H, W, U-1, A)
    For each pixel, the first weight is the weight assigned to the background
    (ie. assigned to not hitting any object).
    The remaining weights are those assigned to some triangles in the scene.
    The attributes measured on those triangles are contained in `attributes`.
    """
    uvs, _, triangle_id_image, depth_image = renderer.rasterize(
        transform[None, ...], vertices, faces, jnp.array([[0, len(faces)]])
    )
    vertices = transform.apply(vertices)

    triangle_intersected_padded = jnp.pad(
        triangle_id_image, pad_width=[(hyperparams.WINDOW, hyperparams.WINDOW)], constant_values=-1
    )

    h = HyperparamsAndIntrinsics(hyperparams, renderer.fx, renderer.fy, renderer.cx, renderer.cy)
    (weights, attributes) = jax.vmap(_get_pixel_attribute_dist_parameters, in_axes=(0, None))(
        b3d.all_pairs(renderer.height, renderer.width),
        (vertices, faces, vertex_attributes, triangle_intersected_padded, h)
    )
    weights = weights.reshape(renderer.height, renderer.width, -1)
    attributes = attributes.reshape(renderer.height, renderer.width, -1, vertex_attributes.shape[1])

    return (weights, attributes)

def render_to_average_rgbd(
    renderer,
    vertices,
    faces,
    vertex_rgbs,
    background_attribute=jnp.array([0.1, 0.1, 0.1, 0]),
    hyperparams=DEFAULT_HYPERPARAMS,
):
    """
    Variant of `render_to_average` for rendering RGBD.
    """
    vertex_depths = vertices[:, 2]
    vertex_rgbds = jnp.concatenate([vertex_rgbs, vertex_depths[:, None]], axis=1)
    return render_to_average(renderer, vertices, faces, vertex_rgbds, background_attribute, hyperparams)

def render_to_average(
        renderer,
        vertices,
        faces,
        vertex_attributes,
        background_attribute,
        hyperparams=DEFAULT_HYPERPARAMS
):
    """
    Differentiable rendering to produce an image by averaging
    the categorical distribution over attributes (e.g. RGB or RGBD)
    returned by `render_to_dist_params`.

    Args:
    - renderer: b3d.Renderer
    - vertices: (V, 3)
    - faces: (F, 3)
    - vertex_attributes: (F, A) [A=3 for RGB; A=4 for RGBD]
    - background_attribute: (A,) attribute to assign to pixels not hitting any object
    - hyperparams: DifferentiableRendererHyperparams
    Returns:
    - image (H, W, A)
    """
    weights, attributes = render_to_dist_params(renderer, vertices, faces, vertex_attributes, hyperparams=hyperparams)
    return dist_params_to_average(weights, attributes, background_attribute)    

def dist_params_to_average(weights, attributes, background_attribute):
    extended_attributes = jnp.concatenate([jnp.tile(background_attribute, (attributes.shape[0], attributes.shape[1], 1, 1)), attributes], axis=2)
    image = jnp.sum(weights[..., None] * extended_attributes, axis=2)
    return image

###############################################
# Core weighting & attribute computation math #
###############################################

def get_pixel_attribute_dist_parameters(
    ij, vertices, faces, vertex_attributes, triangle_intersected_padded,
    hyperparams_and_intrinsics
):
    unique_triangle_indices, weights, barycentric_coords = get_weights_and_barycentric_coords(
        ij, vertices, faces, triangle_intersected_padded, hyperparams_and_intrinsics
    )
    unique_triangle_indices_safe = jnp.where(unique_triangle_indices < 0, 0, unique_triangle_indices)

    # (U - 1, A)
    attributes_at_triangles = jax.vmap(barycentric_interpolation, in_axes=(0, None, 0))(
        faces[unique_triangle_indices_safe[1:]],
        vertex_attributes,
        barycentric_coords
    )

    attributes_at_triangles = jnp.where(
        unique_triangle_indices[1:, None] < 0, -jnp.ones_like(attributes_at_triangles), attributes_at_triangles
    )

    return (weights, attributes_at_triangles)

def _get_pixel_attribute_dist_parameters(ij, args):
    return get_pixel_attribute_dist_parameters(ij, *args)




def get_weights_and_barycentric_coords(ij, vertices, faces, triangle_intersected_padded, hyperparams_and_intrinsics):
    """
    Returns:
    - unique_triangle_indices (U,)
        The values will be
            - one token -10 for the background, at index 0
            - one token -2 of padding (ignore these)
            - the rest are the indices of the unique triangles in the window
    - weights (U,)
        The weights for each triangle.  Will be 0 in every slot where `unique_triangle_indices` is -2.
    - barycentric_coords (U - 1, 3)
        The interpolated attributes for each triangle (and nothing for the background).
        Will filled with -1s for every triangle where `unique_triangle_indices` is -2.
    """
    h = hyperparams_and_intrinsics.hyperparams
    (WINDOW, SIGMA, GAMMA, EPSILON) = h.WINDOW, h.SIGMA, h.GAMMA, h.EPSILON

    triangle_intersected_padded_in_window = jax.lax.dynamic_slice(
        triangle_intersected_padded,
        (ij[0], ij[1]),
        (2 * WINDOW + 1, 2 * WINDOW + 1),
    )
    # This will have the value -2 in slots we should ignore
    # and -1 in slots which hit the background.
    unique_triangle_values = jnp.unique(
        triangle_intersected_padded_in_window, size=triangle_intersected_padded_in_window.size,
        fill_value = -1
    ) - 1
    unique_triangle_values_safe = jnp.where(unique_triangle_values < 0, unique_triangle_values[0], unique_triangle_values)
    
    signed_dist_values, barycentric_coords = get_signed_dists_and_barycentric_coords(
        ij, unique_triangle_values_safe, vertices, faces, hyperparams_and_intrinsics
    )
    z_values = get_z_values(ij, unique_triangle_values_safe, vertices, faces, hyperparams_and_intrinsics)
    z_values = jnp.where(unique_triangle_values >= 0, z_values, z_values.max())
    
    # Math from the softras paper
    signed_dist_scores = jax.nn.sigmoid(jnp.sign(signed_dist_values) * signed_dist_values ** 2 / SIGMA)

    # following https://github.com/kach/softraxterizer/blob/main/softraxterizer.py
    maxz = jnp.where(unique_triangle_values >= 0, z_values, -BIGNUM).max()
    minz = jnp.where(unique_triangle_values >= 0, z_values, BIGNUM).min()
    z = (maxz - z_values) / (maxz - minz + 1e-4)
    zexp = jnp.exp(jnp.clip(z / GAMMA, -20., 20.))

    unnorm_weights = signed_dist_scores * zexp

    # filter out the padding
    unnorm_weights = jnp.where(unique_triangle_values >= 0, unnorm_weights, 0.0)
    unnorm_weights = jnp.concatenate([jnp.array([jnp.exp(jnp.clip(EPSILON/GAMMA, -20, 20))]), unnorm_weights])
    weights = unnorm_weights / jnp.sum(unnorm_weights)
    
    extended_triangle_indices = jnp.concatenate([jnp.array([-10]), unique_triangle_values])
    barycentric_coords = jnp.where(unique_triangle_values[:, None] >= 0, barycentric_coords, -jnp.ones_like(barycentric_coords))
    return (extended_triangle_indices, weights, barycentric_coords)

def barycentric_interpolation(vertex_indices, vertex_attributes, barycentric_coords):
    """
    Args:
    - vertex_indices: (3,) indices of the vertices in the triangle
    - vertex_attributes: (V, A) attributes of the vertices
    - barycentric_coords: (3,) barycentric coordinates of the point
    Returns:
    - interpolated_attributes: (A,) interpolated attributes of the point
    """
    return jnp.dot(barycentric_coords, vertex_attributes[vertex_indices])

################################################################
### Geometric calculations used in the softras-style scoring ###
################################################################

def get_z_values(ij, unique_triangle_values, vertices, faces, hyperparams_and_intrinsics):
    return jax.vmap(get_z_value, in_axes=(None, 0, None, None, None))(
        ij, unique_triangle_values, vertices, faces, hyperparams_and_intrinsics
    )
def get_z_value(ij, triangle_idx, vertices, faces, hyperparams_and_intrinsics):
    """
    Project pixel (i, j) to the plane of `triangle_idx`, then
    compute the z value of the projected point.
    """
    triangle = vertices[faces[triangle_idx]] # 3 x 3 (face_idx, vertex_idx)
    point_on_plane = project_pixel_to_plane(ij, triangle, hyperparams_and_intrinsics)
    return point_on_plane[2]

def get_signed_dists_and_barycentric_coords(ij, unique_triangle_values, vertices, faces, hyperparams_and_intrinsics):
    return jax.vmap(get_signed_dist_and_barycentric_coords, in_axes=(None, 0, None, None, None))(
        ij, unique_triangle_values, vertices, faces, hyperparams_and_intrinsics
    )

def get_signed_dist_and_barycentric_coords(ij, triangle_idx, vertices, faces, hyperparams_and_intrinsics):
    """
    Project pixel (i, j) to the plane of `triangle_idx`, obtaining 3D point `p`,
    then compute
     (1) the signed distance within that plane from p
        to the boundary of the triangle.  (Positive = inside the triangle,
        negative = outside the triangle.)
     (2) the barycentric coordinates of `p` within the triangle, as a triple (a, b, c)
        summing to 1 (up to floating point).  These coordinates give the triangle area across
        from the 0th, 1st, and 2nd vertices in triangle `triangle_idx`, respectively.
    """
    triangle = vertices[faces[triangle_idx]] # 3 x 3 (face_idx, vertex_idx)
    point_on_plane = project_pixel_to_plane(ij, triangle, hyperparams_and_intrinsics)
    
    # distances to 3 lines making up the triangle
    d1 = dist_to_line_seg(triangle[0], triangle[1], point_on_plane)
    d2 = dist_to_line_seg(triangle[1], triangle[2], point_on_plane)
    d3 = dist_to_line_seg(triangle[2], triangle[0], point_on_plane)
    d = jnp.minimum(d1, jnp.minimum(d2, d3))

    a = _signed_area_to_point(triangle[1], triangle[2], point_on_plane)
    b = _signed_area_to_point(triangle[2], triangle[0], point_on_plane)
    c = _signed_area_to_point(triangle[0], triangle[1], point_on_plane)
    in_triangle = jnp.logical_and(
        jnp.equal(jnp.sign(a), jnp.sign(b)),
        jnp.equal(jnp.sign(b), jnp.sign(c))
    )
    signed_distance = jnp.where(in_triangle, d, -d)

    bary = jnp.array([a, b, c]) / (a + b + c + 1e-6)
    bary = jnp.clip(bary, 0., 1.)
    bary = bary / (jnp.sum(bary) + 1e-6)
    
    return (signed_distance, bary)

# From ChatGPT + I fixed a couple bugs in it.
def project_pixel_to_plane(ij, triangle, hyperparams_and_intrinsics):
    """
    Project pixel ij to the plane defined by the given triangle.
    Args:
    - ij: (2,) pixel coordinates
    - triangle: (3, 3) vertices of the triangle (triangle[f] is one vertex)
    """
    fx, fy, cx, cy = hyperparams_and_intrinsics.fx, hyperparams_and_intrinsics.fy, hyperparams_and_intrinsics.cx, hyperparams_and_intrinsics.cy
    y, x = ij
    vertex1, vertex2, vertex3 = triangle

    # Convert pixel coordinates to normalized camera coordinates
    x_c = (x - cx) / fx
    y_c = (y - cy) / fy
    z_c = 1.0  # Assume the camera looks along the +z axis

    # Camera coordinates to the ray direction vector
    ray_dir = jnp.array([x_c, y_c, z_c])
    # jax.debug.print("ray_dir = {rd}", rd=ray_dir)
    # checkify.check(jnp.linalg.norm(ray_dir) > 1e-6, "Ray direction vector {x}", x=ray_dir)
    ray_dir = ray_dir / jnp.linalg.norm(ray_dir)  # Normalize the direction vector

    # Calculate the normal vector of the plane defined by the triangle
    v1_v2 = vertex2 - vertex1
    v1_v3 = vertex3 - vertex1
    normal = jnp.cross(v1_v2, v1_v3)
    normal = normal / jnp.linalg.norm(normal)  # Normalize the normal vector

    # Plane equation: normal . (X - vertex1) = 0
    # Solve for t in the equation: ray_origin + t * ray_dir = X
    # ray_origin is the camera origin, assumed to be at [0, 0, 0]
    # So the equation simplifies to: t * ray_dir = X
    # Substitute in plane equation: normal . (t * ray_dir - vertex1) = 0
    # t = normal . vertex1 / (normal . ray_dir)
    ray_origin = jnp.array([0.0, 0.0, 0.0])
    denom = jnp.dot(normal, ray_dir)
    # if jnp.abs(denom) < 1e-6:
    #     return None  # No intersection if the ray is parallel to the plane

    t = jnp.dot(normal, vertex1 - ray_origin) / (denom + 1e-5)
    intersection_point = ray_origin + t * ray_dir
    
    return jnp.where(
        jnp.abs(denom) < 1e-6, -jnp.ones(3), intersection_point
    )

# The functions below here are following
# https://github.com/kach/softraxterizer/blob/main/softraxterizer.py
def dist_to_line_seg(a, b, p):
    Va = b - a
    Vp = p - a
    projln = Vp.dot(Va) / Va.dot(Va)
    projln = jnp.clip(projln, 0., 1.)
    return jnp.linalg.norm(Vp - projln * Va)

def _signed_area_to_point(a, b, p):
    Va = b - a
    area = jnp.cross(Va, p - a)[2] / 2
    return area