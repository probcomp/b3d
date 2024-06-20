import jax
import jax.numpy as jnp

# From ChatGPT
def tessellate(width, height):
    """
    Tessellate the 2D plane using unit equilateral triangles.
    Will produce a tessellation that fully covers the square region
    with bottom left corner (0, 0) and top right corner (width, height).

    Returns:
    - vertices: (N, 2) array of vertex coordinates
    - faces: (M, 3) array of faces, each row contains the indices of the 3 vertices in a triangle
    """
    num_points_x = width + 3  # Extend by one column on either side
    num_points_y = int(height / (jnp.sqrt(3) / 2)) + 1

    # Generate the x and y coordinates for all points
    x_coords = jnp.arange(num_points_x) - 1  # Shift x coordinates to extend on either side
    y_coords = jnp.arange(num_points_y) * (jnp.sqrt(3) / 2)
    
    # Create a grid of x and y coordinates
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords)
    
    # Apply offset to every other row
    offsets = jnp.arange(num_points_y) % 2 * 0.5
    x_grid = x_grid + offsets[:, None]
    
    # Flatten the grid to get the vertices
    vertices = jnp.vstack((x_grid.ravel(), y_grid.ravel())).T

    # Generate the faces
    def generate_faces(y):
        p0 = y * num_points_x + jnp.arange(num_points_x - 1)
        p1 = p0 + 1
        p2 = p0 + num_points_x
        p3 = p2 + 1
        
        even_row_faces = jnp.stack([p0, p2, p1], axis=-1)
        even_row_faces = jnp.concatenate([even_row_faces, jnp.stack([p1[:-1], p2[:-1], p3[:-1]], axis=-1)], axis=0)
        
        odd_row_faces = jnp.stack([p0, p3, p1], axis=-1)
        odd_row_faces = jnp.concatenate([odd_row_faces, jnp.stack([p0[1:], p2[1:], p3[1:]], axis=-1)], axis=0)
        
        return jax.lax.cond(y % 2 == 0, lambda _: even_row_faces, lambda _: odd_row_faces, None)
    
    faces = jax.vmap(generate_faces)(jnp.arange(num_points_y - 1))
    faces = faces.reshape(-1, 3)

    return vertices, faces


###
def all_pairs_2(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(X, Y), axis=-1),
        0, 1
    ).reshape(-1, 2)

def triangle2D_to_integer_points(triangle, max_step_x, max_step_y):
    """
    Given a triangle = [v1, v2, v3], where each v is a (2,) array,
    return an integer array of size (max_step_x * max_step_y, 2) containing
    all integers (i, j) which fall within the triangle. Fill any additional spots
    with (-10000, -10000).

    This will operate by considering a grid of points of size (max_step_x, max_step_y),
    and checking if each point in this grid is within the triangle.
    Setting these values too small means some points will be missed; setting them
    to be large will increase the amount of performed computation.
    These should be set to the smallest possible values where it is guaranteed
    all integral coordinates falling within `triangle` fall in a max_step_x x max_step_y grid.
    """
    v1, v2, v3 = triangle
    triangle = jnp.array([[v1[1], v1[0]], [v2[1], v2[0]], [v3[1], v3[0]]])
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    def is_point_in_triangle(points, tri):
        v1, v2, v3 = tri
        b1 = sign(points, v1, v2) < 1e-3
        b2 = sign(points, v2, v3) < 1e-3
        b3 = sign(points, v3, v1) < 1e-3
        return (b1 == b2) & (b2 == b3)

    # Bounding box for the triangle
    min_x = jnp.floor(jnp.min(triangle[:, 0])).astype(int)
    min_y = jnp.floor(jnp.min(triangle[:, 1])).astype(int)
    max_x = jnp.ceil(jnp.max(triangle[:, 0])).astype(int)
    max_y = jnp.ceil(jnp.max(triangle[:, 1])).astype(int)

    # Generate all integer points within the bounding box
    x_coords = min_x + jnp.arange(0, max_step_x)
    y_coords = min_y + jnp.arange(0, max_step_y)
    points = all_pairs_2(x_coords, y_coords).T

    # Filter points that are inside the triangle
    inside_mask = is_point_in_triangle(points, triangle)
    points = jnp.where(
        inside_mask[None, :],
        points,
        jnp.array([-10000, -10000])[:, None]
    )
    return points

def triangle2D_to_pixel_coords(triangle, max_step_x, max_step_y):
    """
    Identify which pixel coordinates land within this triangle.
    (These are points (i + 0.5, j + 0.5) on the plane.)

    `max_step_x` and `max_step_y` upper bound the size of the JAX computation,
    to enable static shapes.
    They control the size of the grid of pixels which are considered as possibly
    intersecting the triangle.
    These should be set to the smallest possible values where it is guaranteed
    all pixels falling within `triangle` fall in a max_step_x x max_step_y grid.
    """
    return triangle2D_to_integer_points(triangle - 0.5, max_step_x, max_step_y)

def triangle2D_to_color(triangle, rgb_image, max_step_x, max_step_y):
    """
    Given a 2D triangle (array of shape (3, 2)), and an RGB image where
    each pixel corresponds to a color measured at point (i + 0.5, j + 0.5)
    on the plane, return a color within each triangle.
    If no pixels are within a triangle, return black.

    `max_step_x` and `max_step_y` upper bound the size of the JAX computation,
    to enable static shapes.
    They control the size of the grid of pixels which are considered as possibly
    intersecting the triangle.
    These should be set to the smallest possible values where it is guaranteed
    all pixels falling within `triangle` fall in a max_step_x x max_step_y grid.
    """
    pixel_coords = triangle2D_to_pixel_coords(triangle, max_step_x, max_step_y)

    # get first element of pixel_coords with value not -10000
    first_nonnegative = jnp.argmax(pixel_coords[0] != -10000)
    ij = pixel_coords[:, first_nonnegative]
    i_safe = jnp.clip(ij[0], 0, rgb_image.shape[0] - 1)
    j_safe = jnp.clip(ij[1], 0, rgb_image.shape[1] - 1)
    return jnp.where(
        ij[0] == -10000,
        jnp.zeros_like(rgb_image[0, 0]),
        rgb_image[i_safe, j_safe]
    )

def generate_tessellated_2D_mesh_from_rgb_image(rgb_image, scaledown=1):
    """
    Tesselate the plane with equilateral triangles and color each triangle
    based on the provided RGB image.
    The equilateral triangles will have side length 3.0, and will have
    the same color as one of the pixel centers in the RGB image that lands
    within that triangle.

    Args:
    - rgb_image

    Returns:
    - vertices: (V, 2) array of 2D vertices
    - faces: (F, 3) array of faces, each row contains the indices of the 3 vertices in a triangle
    - triangle_colors: (F, 3) array of colors for each triangle
    """
    height, width, _ = rgb_image.shape
    div = 3 * scaledown
    vertices, faces = tessellate(width // div, height // div)
    vertices = vertices * div
    triangle_colors = jax.vmap(triangle2D_to_color, in_axes=(0, None, None, None))(vertices[faces], rgb_image, 5 * scaledown, 5 * scaledown)
    return (vertices, faces, triangle_colors)