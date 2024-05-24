import jax
import jax.numpy as jnp
import genjax
import rerun as rr
import b3d
import os

# From ChatGPT
def tessellate(width, height):
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

def separate_shared_vertices(vertices, faces):
    """
    Given a mesh where multiple faces are using the same vertex,
    return a mesh where each vertex is unique to a face.
    (This will therefore duplicate some vertices.)
    """
    # Flatten the faces array and use it to index into the vertices array
    flat_faces = faces.ravel()
    unique_vertices = vertices[flat_faces]

    # Reshape the unique_vertices array to match the faces structure
    new_faces = jnp.arange(unique_vertices.shape[0]).reshape(faces.shape)

    return unique_vertices, new_faces


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
    """
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
    return triangle2D_to_integer_points(triangle - 0.5, max_step_x, max_step_y)

def triangle2D_to_color(triangle, rgb_image, max_step_x, max_step_y):
    pixel_coords = triangle2D_to_pixel_coords(triangle, max_step_x, max_step_y)

    # get first element of pixel_coords with value not -10000
    first_nonnegative = jnp.argmax(pixel_coords[0] != -10000)
    ij = pixel_coords[:, first_nonnegative]
    i_safe = jnp.clip(ij[0], 0, rgb_image.shape[0] - 1)
    j_safe = jnp.clip(ij[1], 0, rgb_image.shape[1] - 1)
    return jnp.where(
        ij[0] == -10000,
        jnp.zeros(3),
        rgb_image[i_safe, j_safe]
    )


v, f = tessellate(20, 20)
pixel_coords = triangle2D_to_pixel_coords(v[f[4]], 2, 2)

rgb_image = jnp.ones((100, 120, 3))
c = triangle2D_to_color(v[f[4]], rgb_image, 2, 2)

###

def generate_initial_mesh(key, rgb_image, mindepth, maxdepth, focal_length):
    # Get the dimensions of the image
    height, width, _ = rgb_image.shape
    height, width = height // 3, width // 3

    # Generate the tessellated mesh
    vertices_2D, faces = tessellate(width, height)
    vertices_2D = vertices_2D * 3.0

    # vertex_colors = genjax.uniform.sample(key, jnp.zeros((vertices.shape[0], 3)), jnp.ones((vertices.shape[0], 3)))
    triangle_colors = jax.vmap(triangle2D_to_color, in_axes=(0, None, None, None))(vertices_2D[faces], rgb_image, 5, 5)
    #genjax.uniform.sample(key, jnp.zeros((faces.shape[0], 3)), jnp.ones((faces.shape[0], 3)))

    depth = genjax.uniform.sample(key, mindepth, maxdepth)
    vertices = jnp.hstack((vertices_2D * depth / focal_length, jnp.ones((vertices_2D.shape[0], 1)) * depth))

    vertices_2, faces_2 = separate_shared_vertices(vertices, faces)
    vertex_colors_2 = jnp.repeat(triangle_colors, 3, axis=0)

    return (vertices_2, faces_2, vertex_colors_2)

rr.init("tessellation")
rr.connect("127.0.0.1:8812")

key = jax.random.PRNGKey(0)
(v, f, vc) = generate_initial_mesh(key, jnp.ones((100, 120, 3)), 0.1, 3.0, 64.0)

rr.log("my_mesh", rr.Mesh3D(
    vertex_positions=v,
    indices=f,
    vertex_colors=vc
))

#####

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)
image_width, image_height, fx, fy, cx, cy, near, far = jnp.array(
    video_input.camera_intrinsics_depth
)

rgbs = video_input.rgb[::4] / 255.0

(v, f, vc) = generate_initial_mesh(key, rgbs[0], 0.1, 3.0, fx)
rr.log("my_mesh", rr.Mesh3D(
    vertex_positions=v,
    indices=f,
    vertex_colors=vc
))