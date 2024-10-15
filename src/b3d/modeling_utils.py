import itertools

import genjax
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from b3d.pose import (
    logpdf_gaussian_vmf_pose,
    logpdf_uniform_pose,
    logpdf_uniform_scale,
    sample_gaussian_vmf_pose,
    sample_uniform_pose,
    sample_uniform_scale,
)

############################
###### gjk algorithm #######
############################


# Support function: finds the farthest point in a given direction
def support_function(vertices1, vertices2, direction):
    point1 = vertices1[jnp.argmax(jnp.dot(vertices1, direction))]
    point2 = vertices2[jnp.argmax(jnp.dot(vertices2, -direction))]
    return point1 - point2


# Check if the origin is in the simplex formed by the given points
def contains_origin(simplex, direction):
    if len(simplex) == 2:
        a, b = simplex
        ab = b - a
        ao = -a

        def true_fn(_):
            new_direction = jnp.cross(jnp.cross(ab, ao), ab)
            return False, simplex, new_direction

        def false_fn(_):
            return False, [a, None, None], ao  # Pad with None

        return jax.lax.cond(jnp.dot(ab, ao) > 0, true_fn, false_fn, None)

    elif len(simplex) == 3:
        a, b, c = simplex
        ab = b - a
        ac = c - a
        ao = -a
        abc = jnp.cross(ab, ac)

        def cond1(_):
            def cond1_true(_):
                new_direction = jnp.cross(jnp.cross(ac, ao), ac)
                return False, [a, c, None], new_direction  # Pad with None

            def cond1_false(_):
                return False, [a, None, None], ao  # Pad with None

            return jax.lax.cond(jnp.dot(ac, ao) > 0, cond1_true, cond1_false, None)

        def cond2(_):
            def cond2_true(_):
                new_direction = jnp.cross(jnp.cross(ab, ao), ab)
                return False, [a, b, None], new_direction  # Pad with None

            def cond2_false(_):
                def cond3_true(_):
                    return False, [a, b, c], abc

                def cond3_false(_):
                    return False, [a, c, b], -abc

                return jax.lax.cond(jnp.dot(abc, ao) > 0, cond3_true, cond3_false, None)

            return jax.lax.cond(
                jnp.dot(jnp.cross(ab, abc), ao) > 0, cond2_true, cond2_false, None
            )

        return jax.lax.cond(jnp.dot(jnp.cross(abc, ac), ao) > 0, cond1, cond2, None)

    return True, simplex, direction


# Main GJK algorithm using jax.lax.cond
def gjk(vertices1, vertices2):
    direction = jnp.array([1.0, 0.0, 0.0])
    simplex = [
        support_function(vertices1, vertices2, direction),
        None,
        None,
    ]  # Start with a fixed-size simplex list

    direction = -simplex[0]

    def gjk_step(i, data):
        simplex, direction, collision_detected = data
        new_point = support_function(vertices1, vertices2, direction)

        def true_fn(_):
            updated_simplex = [new_point if item is None else item for item in simplex]
            return updated_simplex

        def false_fn(_):
            return simplex

        # Ensure `simplex` is always the same size
        simplex = jax.lax.cond(collision_detected, true_fn, false_fn, None)

        # Check if the origin is contained in the updated simplex
        if collision_detected:
            is_collision, simplex, direction = contains_origin(simplex, direction)
            collision_detected = is_collision

        return simplex, direction, collision_detected

    # Initialize the loop
    simplex, direction, collision_detected = jax.lax.fori_loop(
        0, 20, gjk_step, (simplex, direction, True)
    )

    return collision_detected


def get_interpenetration(mesh_seq):
    for pair in list(itertools.combinations(mesh_seq, 2)):
        m1, m2 = pair
        # Compute intersection volume
        intersect = gjk(m1.vertices, m2.vertices)
        if intersect:
            return True
    return False


############################
#### compute the volume ####
############################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     epsilon = 1e-6
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     parallel = jnp.abs(a) < epsilon
#     f = 1.0 / a
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     valid_u = (u >= 0.0) & (u <= 1.0)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     valid_v = (v >= 0.0) & (u + v <= 1.0)
#     t = f * jnp.dot(e2, q)
#     valid_t = t > epsilon
#     intersects = (~parallel) & valid_u & valid_v & valid_t
#     return intersects


# @jax.jit
# def point_in_mesh(point, vertices, faces):
#     ray_direction = jnp.array([1.0, 0.0, 0.0])  # Arbitrary direction
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]

#     intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
#         point, ray_direction, v0, v1, v2
#     )
#     num_intersections = jnp.sum(intersects)
#     return num_intersections % 2 == 1  # Inside if odd number of intersections


# def min_max_coord(vertices):
#     min_coords = jnp.min(vertices, axis=0)
#     max_coords = jnp.max(vertices, axis=0)
#     return min_coords, max_coords


# @partial(jax.jit, static_argnames=["num_samples"])
# def monte_carlo_intersection_volume(
#     mesh1_vertices, mesh1_faces, mesh2_vertices, mesh2_faces, num_samples, key
# ):
#     min_coords1, max_coords1 = min_max_coord(mesh1_vertices)
#     min_coords2, max_coords2 = min_max_coord(mesh2_vertices)

#     min_coords = jnp.maximum(min_coords1, min_coords2)
#     max_coords = jnp.minimum(max_coords1, max_coords2)

#     overlap = jnp.all(min_coords < max_coords)
#     bbox_volume = jnp.prod(max_coords - min_coords)

#     def sample_points(key):
#         subkey_x, subkey_y, subkey_z = jax.random.split(key, 3)
#         x = jax.random.uniform(
#             subkey_x, shape=(num_samples,), minval=min_coords[0], maxval=max_coords[0]
#         )
#         y = jax.random.uniform(
#             subkey_y, shape=(num_samples,), minval=min_coords[1], maxval=max_coords[1]
#         )
#         z = jax.random.uniform(
#             subkey_z, shape=(num_samples,), minval=min_coords[2], maxval=max_coords[2]
#         )
#         points = jnp.stack([x, y, z], axis=1)
#         return points

#     points = sample_points(key)

#     point_in_mesh_vmap = jax.vmap(point_in_mesh, in_axes=(0, None, None))

#     in_mesh1 = point_in_mesh_vmap(points, mesh1_vertices, mesh1_faces)
#     in_mesh2 = point_in_mesh_vmap(points, mesh2_vertices, mesh2_faces)
#     in_both_meshes = in_mesh1 & in_mesh2

#     hits = jnp.sum(in_both_meshes)
#     intersection_volume = (hits / num_samples) * bbox_volume * overlap
#     return intersection_volume


# def get_interpenetration(mesh_seq, num_samples):
#     interpenetrations = []
#     for ct, pair in enumerate(list(itertools.combinations(mesh_seq, 2))):
#         m1, m2 = pair
#         # Monte Carlo parameters
#         key = jax.random.PRNGKey(ct)  # Random seed
#         # Compute intersection volume
#         intersection_volume = monte_carlo_intersection_volume(
#             m1.vertices, m1.faces, m2.vertices, m2.faces, num_samples, key
#         )
#         interpenetrations.append(intersection_volume)
#     return jnp.array(interpenetrations).sum()


####################################################
#### memory-efficient way to compute the volume ####
####################################################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     epsilon = 1e-8
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     parallel = jnp.abs(a) < epsilon
#     f = jnp.where(~parallel, 1.0 / a, 0.0)
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     t = f * jnp.dot(e2, q)
#     intersects = (~parallel) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)
#     return intersects


# @jax.jit
# def point_in_mesh(point, vertices, faces):
#     ray_direction = jnp.array([1.0, 0.0, 0.0])
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]

#     def intersect_triangle(carry, i):
#         intersects = ray_intersects_triangle(point, ray_direction, v0[i], v1[i], v2[i])
#         count = carry + intersects
#         return count, None

#     num_faces = faces.shape[0]
#     total_intersections, _ = jax.lax.scan(intersect_triangle, 0, jnp.arange(num_faces))
#     inside = total_intersections % 2 == 1
#     return inside


# @jax.jit
# def process_batch(points, mesh1_vertices, mesh1_faces, mesh2_vertices, mesh2_faces):
#     in_mesh1 = jax.vmap(point_in_mesh, in_axes=(0, None, None))(
#         points, mesh1_vertices, mesh1_faces
#     )
#     in_mesh2 = jax.vmap(point_in_mesh, in_axes=(0, None, None))(
#         points, mesh2_vertices, mesh2_faces
#     )
#     in_both = in_mesh1 & in_mesh2
#     return in_both


# def monte_carlo_interpenetration_volume(
#     mesh1, mesh2, num_samples, key, batch_size=100000
# ):
#     # Compute overlapping bounding box using JAX operations
#     min_coords1 = jnp.min(mesh1.vertices, axis=0)
#     max_coords1 = jnp.max(mesh1.vertices, axis=0)
#     min_coords2 = jnp.min(mesh2.vertices, axis=0)
#     max_coords2 = jnp.max(mesh2.vertices, axis=0)

#     min_coords = jnp.maximum(min_coords1, min_coords2)
#     max_coords = jnp.minimum(max_coords1, max_coords2)

#     # Compute overlap using JAX operations
#     overlap = jnp.all(min_coords < max_coords)

#     # Use lax.cond to handle control flow based on JAX arrays
#     def compute_volume(_):
#         bbox_volume = jnp.prod(max_coords - min_coords)

#         # Generate random points
#         num_full_batches = num_samples // batch_size
#         remainder = num_samples % batch_size
#         total_batches = num_full_batches + (1 if remainder > 0 else 0)

#         keys = jax.random.split(key, num=total_batches)

#         # Initialize total hits
#         total_hits = 0

#         for i in range(total_batches):
#             subkey = keys[i]
#             current_batch_size = batch_size if i < num_full_batches else remainder

#             points = jax.random.uniform(
#                 subkey,
#                 shape=(current_batch_size, 3),
#                 minval=min_coords,
#                 maxval=max_coords,
#             )

#             # Pad the last batch if necessary
#             if current_batch_size < batch_size:
#                 pad_size = batch_size - current_batch_size
#                 points = jnp.pad(points, ((0, pad_size), (0, 0)), mode="constant")

#             # Perform point-in-mesh tests
#             in_both = process_batch(
#                 points, mesh1.vertices, mesh1.faces, mesh2.vertices, mesh2.faces
#             )

#             # Mask out the padded points
#             if current_batch_size < batch_size:
#                 mask = jnp.arange(batch_size) < current_batch_size
#                 in_both = in_both & mask

#             hits = jnp.sum(in_both[:current_batch_size])
#             total_hits += hits

#         interpenetration_volume = (total_hits / num_samples) * bbox_volume
#         return interpenetration_volume

#     # Function to return zero volume
#     def zero_volume(_):
#         return 0.0

#     # Use lax.cond to choose between computing volume or returning zero
#     interpenetration_volume = jax.lax.cond(
#         overlap, compute_volume, zero_volume, operand=None
#     )

#     return interpenetration_volume


# def get_interpenetration(mesh_seq, num_samples):
#     interpenetrations = []
#     for ct, pair in enumerate(list(itertools.combinations(mesh_seq, 2))):
#         m1, m2 = pair
#         # Monte Carlo parameters
#         key = jax.random.PRNGKey(ct)  # Random seed
#         # Compute intersection volume
#         intersection_volume = monte_carlo_interpenetration_volume(
#             m1, m2, num_samples, key
#         )
#         interpenetrations.append(intersection_volume)
#     return jnp.array(interpenetrations).sum()


#############################################################
#### the most memory-efficient way to compute the volume ####
#############################################################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     epsilon = 1e-8
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     f = 1.0 / a
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     t = f * jnp.dot(e2, q)
#     cond = (jnp.abs(a) > epsilon) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)
#     return cond

# @jax.jit
# def process_face_batch(point, ray_direction, v0, v1, v2):
#     intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
#         point, ray_direction, v0, v1, v2
#     )
#     batch_count = jnp.sum(intersects)
#     return batch_count

# def point_in_mesh(point, vertices, faces, batch_size=4096):
#     num_faces = faces.shape[0]
#     num_batches = (num_faces + batch_size - 1) // batch_size

#     total_intersections = 0
#     ray_direction = jnp.array([1.0, 0.0, 0.0])

#     for i in range(num_batches):
#         start = i * batch_size
#         end = min(start + batch_size, num_faces)
#         batch_faces = faces[start:end]
#         v0 = vertices[batch_faces[:, 0]]
#         v1 = vertices[batch_faces[:, 1]]
#         v2 = vertices[batch_faces[:, 2]]

#         batch_count = process_face_batch(point, ray_direction, v0, v1, v2)
#         total_intersections += batch_count

#     inside = total_intersections % 2 == 1
#     return inside

# def process_batch(points, mesh1, mesh2, batch_size_faces=4096):
#     in_mesh1 = jax.vmap(lambda p: point_in_mesh(p, mesh1.vertices, mesh1.faces, batch_size_faces))(points)
#     in_mesh2 = jax.vmap(lambda p: point_in_mesh(p, mesh2.vertices, mesh2.faces, batch_size_faces))(points)
#     in_both = in_mesh1 & in_mesh2
#     return in_both

# def monte_carlo_interpenetration_volume(mesh1, mesh2, num_samples, key1, batch_size_points=10000, batch_size_faces=4096):
#     min_coords1 = jnp.min(mesh1.vertices, axis=0)
#     max_coords1 = jnp.max(mesh1.vertices, axis=0)
#     min_coords2 = jnp.min(mesh2.vertices, axis=0)
#     max_coords2 = jnp.max(mesh2.vertices, axis=0)

#     min_coords = jnp.maximum(min_coords1, min_coords2)
#     max_coords = jnp.minimum(max_coords1, max_coords2)

#     overlap = jnp.all(min_coords < max_coords)

#     def compute_volume(_):
#         bbox_volume = jnp.prod(max_coords - min_coords)
#         num_batches = (num_samples + batch_size_points - 1) // batch_size_points

#         total_hits = 0

#         for i in range(num_batches):
#             key, subkey = jax.random.split(key1)
#             current_batch_size = min(batch_size_points, num_samples - i * batch_size_points)

#             points = jax.random.uniform(
#                 subkey, shape=(current_batch_size, 3), minval=min_coords, maxval=max_coords
#             )

#             # Pad points to fixed batch size if necessary
#             if current_batch_size < batch_size_points:
#                 pad_size = batch_size_points - current_batch_size
#                 points = jnp.pad(points, ((0, pad_size), (0, 0)), mode='edge')

#             # Process the batch
#             in_both = process_batch(points, mesh1, mesh2, batch_size_faces)

#             # Mask out extra points
#             if current_batch_size < batch_size_points:
#                 in_both = in_both[:current_batch_size]

#             hits = jnp.sum(in_both)
#             total_hits += hits

#         interpenetration_volume = (total_hits / num_samples) * bbox_volume
#         return interpenetration_volume

#     interpenetration_volume = jax.lax.cond(
#         overlap,
#         compute_volume,
#         lambda _: 0.0,
#         operand=None
#     )

#     return interpenetration_volume

# def get_interpenetration(mesh_seq, num_samples):
#     interpenetrations = []
#     for ct, pair in enumerate(list(itertools.combinations(mesh_seq, 2))):
#         m1, m2 = pair
#         # Monte Carlo parameters
#         key = jax.random.PRNGKey(ct)  # Random seed
#         # Compute intersection volume
#         intersection_volume = monte_carlo_interpenetration_volume(
#             m1, m2, num_samples, key
#         )
#         interpenetrations.append(intersection_volume)
#     return jnp.array(interpenetrations).sum()


##############################
#### compute the distance ####
##############################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     epsilon = 1e-6
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     parallel = jnp.abs(a) < epsilon
#     f = 1.0 / a
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     t = f * jnp.dot(e2, q)
#     intersects = (~parallel) & (u >= 0.0) & (u <= 1.0) & \
#                  (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)
#     return intersects


# @jax.jit
# def point_in_mesh(point, vertices, faces):
#     ray_direction = jnp.array([1.0, 0.0, 0.0])
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]
#     intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
#         point, ray_direction, v0, v1, v2)
#     num_intersections = jnp.sum(intersects)
#     return num_intersections % 2 == 1


# def get_inside_mask(mesh_points, other_mesh_vertices, other_mesh_faces):
#     point_in_mesh_vmap = jax.vmap(point_in_mesh, in_axes=(0, None, None))
#     inside = point_in_mesh_vmap(mesh_points, other_mesh_vertices, other_mesh_faces)
#     return inside  # Return boolean mask


# @jax.jit
# def point_triangle_distance(p, v0, v1, v2):
#     # ... [Corrected and vectorized implementation]
#     ab = v1 - v0  # (T, 3)
#     ac = v2 - v0
#     ap = p - v0   # p is (3,), broadcasted to (T, 3)

#     # Compute normal
#     n = jnp.cross(ab, ac)
#     n_norm = jnp.linalg.norm(n, axis=1)
#     n = n / n_norm[:, None]

#     # Distance from point to plane
#     dist_to_plane = jnp.abs(jnp.sum(ap * n, axis=1))

#     # Project point onto plane
#     proj_p = p - dist_to_plane[:, None] * n  # (T, 3)

#     # Compute barycentric coordinates
#     vp = proj_p - v0
#     d00 = jnp.sum(ab * ab, axis=1)
#     d01 = jnp.sum(ab * ac, axis=1)
#     d11 = jnp.sum(ac * ac, axis=1)
#     d20 = jnp.sum(vp * ab, axis=1)
#     d21 = jnp.sum(vp * ac, axis=1)
#     denom = d00 * d11 - d01 * d01
#     denom = jnp.where(denom == 0, 1e-8, denom)  # Avoid division by zero

#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w

#     is_inside = (u >= 0) & (v >= 0) & (w >= 0)

#     # Closest point
#     v_clamped = jnp.clip(v, 0.0, 1.0)
#     w_clamped = jnp.clip(w, 0.0, 1.0)
#     u_clamped = 1.0 - v_clamped - w_clamped
#     closest_point = jnp.where(
#         is_inside[:, None],
#         proj_p,
#         u_clamped[:, None] * v0 + v_clamped[:, None] * v1 + w_clamped[:, None] * v2
#     )

#     distance = jnp.linalg.norm(p - closest_point, axis=1)
#     return distance  # (T,)


# def compute_min_distances(points, mesh_vertices, mesh_faces):
#     v0 = mesh_vertices[mesh_faces[:, 0]]  # (T, 3)
#     v1 = mesh_vertices[mesh_faces[:, 1]]
#     v2 = mesh_vertices[mesh_faces[:, 2]]

#     def point_to_mesh_distance(p):
#         distances = point_triangle_distance(p, v0, v1, v2)  # (T,)
#         min_distance = jnp.min(distances)
#         return min_distance

#     distances = jax.vmap(point_to_mesh_distance)(points)  # (P,)
#     return distances


# def maximum_interpenetration_distance(mesh1, mesh2):
#     # Get inside masks
#     inside_mask1 = get_inside_mask(mesh1.vertices, mesh2.vertices, mesh2.faces)
#     inside_mask2 = get_inside_mask(mesh2.vertices, mesh1.vertices, mesh1.faces)

#     # Compute distances
#     distances1 = compute_min_distances(mesh1.vertices, mesh2.vertices, mesh2.faces)
#     distances2 = compute_min_distances(mesh2.vertices, mesh1.vertices, mesh1.faces)

#     # Mask distances for points outside
#     distances1 = jnp.where(inside_mask1, distances1, -jnp.inf)
#     distances2 = jnp.where(inside_mask2, distances2, -jnp.inf)

#     # Compute maximum interpenetration distance
#     max_distance1 = jnp.max(distances1)
#     max_distance2 = jnp.max(distances2)
#     max_interpenetration_distance = jnp.maximum(max_distance1, max_distance2)
#     return max_interpenetration_distance


# def get_interpenetration(mesh_seq, num_sample):
#     interpenetrations = []
#     for pair in list(itertools.combinations(mesh_seq, 2)):
#         m1, m2 = pair
#         # Compute intersection distance
#         intersection_dist = maximum_interpenetration_distance(
#             m1, m2
#         )
#         interpenetrations.append(intersection_dist)
#     return jnp.array(interpenetrations).sum()


########################################################
#### memory-efficient way of computing the distance ####
########################################################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     # Implementation remains the same
#     epsilon = 1e-6
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     parallel = jnp.abs(a) < epsilon
#     f = 1.0 / a
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     t = f * jnp.dot(e2, q)
#     intersects = (~parallel) & (u >= 0.0) & (u <= 1.0) & \
#                  (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)
#     return intersects

# @jax.jit
# def point_in_mesh(point, vertices, faces):
#     ray_direction = jnp.array([1.0, 0.0, 0.0])
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]
#     intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
#         point, ray_direction, v0, v1, v2)
#     num_intersections = jnp.sum(intersects)
#     return num_intersections % 2 == 1

# def get_inside_mask(mesh_points, other_mesh_vertices, other_mesh_faces):
#     point_in_mesh_vmap = jax.vmap(point_in_mesh, in_axes=(0, None, None))
#     inside = point_in_mesh_vmap(mesh_points, other_mesh_vertices, other_mesh_faces)
#     return inside  # Return boolean mask

# def compute_min_distances(points, mesh_vertices, mesh_faces, batch_size=1024):
#     num_points = points.shape[0]
#     distances = []

#     for i in range(0, num_points, batch_size):
#         batch_points = points[i:i+batch_size]

#         # Compute distances for the current batch
#         batch_distances = compute_min_distances_batch(batch_points, mesh_vertices, mesh_faces)
#         distances.append(batch_distances)

#     # Concatenate distances from all batches
#     distances = jnp.concatenate(distances)
#     return distances

# @jax.jit
# def compute_min_distances_batch(points_batch, mesh_vertices, mesh_faces):
#     v0 = mesh_vertices[mesh_faces[:, 0]]  # (T, 3)
#     v1 = mesh_vertices[mesh_faces[:, 1]]
#     v2 = mesh_vertices[mesh_faces[:, 2]]

#     # Function to compute distances for a single point
#     def point_to_mesh_distance(p):
#         distances = point_triangle_distance(p, v0, v1, v2)  # (T,)
#         min_distance = jnp.min(distances)
#         return min_distance

#     # Vectorize over points in the batch
#     distances = jax.vmap(point_to_mesh_distance)(points_batch)  # (batch_size,)
#     return distances

# # def compute_min_distances(points, mesh_vertices, mesh_faces):
# #     v0 = mesh_vertices[mesh_faces[:, 0]]  # (T, 3)
# #     v1 = mesh_vertices[mesh_faces[:, 1]]
# #     v2 = mesh_vertices[mesh_faces[:, 2]]

# #     # Function to compute minimum distance for a single point
# #     @jax.jit
# #     def point_to_mesh_distance(p):
# #         # Use jax.lax.scan to iteratively compute minimum distance
# #         def body_fun(carry, triangle):
# #             v0_t, v1_t, v2_t = triangle
# #             dist = point_triangle_distance(p, v0_t[None, :], v1_t[None, :], v2_t[None, :])[0]
# #             min_dist = jnp.minimum(carry, dist)
# #             return min_dist, None

# #         init = jnp.inf
# #         triangles = (v0, v1, v2)
# #         min_distance, _ = jax.lax.scan(body_fun, init, triangles, length=v0.shape[0])
# #         return min_distance

# #     # Vectorize over points
# #     distances = jax.vmap(point_to_mesh_distance)(points)
# #     return distances

# def maximum_interpenetration_distance(mesh1, mesh2, batch_size=1024):
#     # Get inside masks
#     inside_mask1 = get_inside_mask(mesh1.vertices, mesh2.vertices, mesh2.faces)
#     inside_mask2 = get_inside_mask(mesh2.vertices, mesh1.vertices, mesh1.faces)

#     # Compute distances in batches
#     distances1 = compute_min_distances(mesh1.vertices, mesh2.vertices, mesh2.faces, batch_size)
#     distances2 = compute_min_distances(mesh2.vertices, mesh1.vertices, mesh1.faces, batch_size)

#     # Mask distances for points outside
#     distances1 = jnp.where(inside_mask1, distances1, -jnp.inf)
#     distances2 = jnp.where(inside_mask2, distances2, -jnp.inf)

#     # Compute maximum interpenetration distance
#     max_distance1 = jnp.max(distances1)
#     max_distance2 = jnp.max(distances2)
#     max_interpenetration_distance = jnp.maximum(max_distance1, max_distance2)
#     return max_interpenetration_distance

# @jax.jit
# def point_triangle_distance(p, v0, v1, v2):
#     # Compute edges
#     ab = v1 - v0  # (T, 3)
#     ac = v2 - v0
#     ap = p - v0   # p is (3,), broadcasted to (T, 3)

#     # Compute normal
#     n = jnp.cross(ab, ac)
#     n_norm = jnp.linalg.norm(n, axis=1)
#     n_norm = jnp.where(n_norm == 0, 1e-8, n_norm)  # Avoid division by zero
#     n = n / n_norm[:, None]

#     # Distance from point to plane
#     dist_to_plane = jnp.abs(jnp.sum(ap * n, axis=1))

#     # Project point onto plane
#     proj_p = p - dist_to_plane[:, None] * n  # (T, 3)

#     # Compute barycentric coordinates
#     vp = proj_p - v0
#     d00 = jnp.sum(ab * ab, axis=1)
#     d01 = jnp.sum(ab * ac, axis=1)
#     d11 = jnp.sum(ac * ac, axis=1)
#     d20 = jnp.sum(vp * ab, axis=1)
#     d21 = jnp.sum(vp * ac, axis=1)
#     denom = d00 * d11 - d01 * d01
#     denom = jnp.where(denom == 0, 1e-8, denom)  # Avoid division by zero

#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w

#     is_inside = (u >= 0) & (v >= 0) & (w >= 0)

#     # Closest point
#     v_clamped = jnp.clip(v, 0.0, 1.0)
#     w_clamped = jnp.clip(w, 0.0, 1.0)
#     u_clamped = 1.0 - v_clamped - w_clamped
#     closest_point = jnp.where(
#         is_inside[:, None],
#         proj_p,
#         u_clamped[:, None] * v0 + v_clamped[:, None] * v1 + w_clamped[:, None] * v2
#     )

#     distance = jnp.linalg.norm(p - closest_point, axis=1)
#     return distance  # (T,)


# def get_interpenetration(mesh_seq, num_sample):
#     interpenetrations = []
#     for pair in list(itertools.combinations(mesh_seq, 2)):
#         m1, m2 = pair
#         # Compute intersection distance
#         intersection_dist = maximum_interpenetration_distance(
#             m1, m2
#         )
#         interpenetrations.append(intersection_dist)
#     return jnp.array(interpenetrations).sum()


#################################################################
#### the most memory-efficient way of computing the distance ####
#################################################################
# @jax.jit
# def ray_intersects_triangle(p0, d, v0, v1, v2):
#     epsilon = 1e-6
#     e1 = v1 - v0
#     e2 = v2 - v0
#     h = jnp.cross(d, e2)
#     a = jnp.dot(e1, h)
#     parallel = jnp.abs(a) < epsilon
#     f = 1.0 / a
#     s = p0 - v0
#     u = f * jnp.dot(s, h)
#     q = jnp.cross(s, e1)
#     v = f * jnp.dot(d, q)
#     t = f * jnp.dot(e2, q)
#     intersects = (~parallel) & (u >= 0.0) & (u <= 1.0) & \
#                  (v >= 0.0) & (u + v <= 1.0) & (t > epsilon)
#     return intersects

# @jax.jit
# def point_in_mesh(point, vertices, faces):
#     ray_direction = jnp.array([1.0, 0.0, 0.0])
#     v0 = vertices[faces[:, 0]]
#     v1 = vertices[faces[:, 1]]
#     v2 = vertices[faces[:, 2]]
#     intersects = jax.vmap(ray_intersects_triangle, in_axes=(None, None, 0, 0, 0))(
#         point, ray_direction, v0, v1, v2)
#     num_intersections = jnp.sum(intersects)
#     return num_intersections % 2 == 1

# def get_inside_mask(mesh_points, other_mesh_vertices, other_mesh_faces):
#     point_in_mesh_vmap = jax.vmap(point_in_mesh, in_axes=(0, None, None))
#     inside = point_in_mesh_vmap(mesh_points, other_mesh_vertices, other_mesh_faces)
#     return inside

# @jax.jit
# def point_triangle_distance(p, v0, v1, v2):
#     # Compute edges
#     ab = v1 - v0
#     ac = v2 - v0
#     ap = p - v0

#     # Compute normal
#     n = jnp.cross(ab, ac)
#     n_norm = jnp.linalg.norm(n)
#     n_norm = jnp.where(n_norm == 0, 1e-8, n_norm)  # Avoid division by zero
#     n = n / n_norm

#     # Distance from point to plane
#     dist_to_plane = jnp.abs(jnp.dot(ap, n))

#     # Project point onto plane
#     proj_p = p - dist_to_plane * n

#     # Compute barycentric coordinates
#     vp = proj_p - v0
#     d00 = jnp.dot(ab, ab)
#     d01 = jnp.dot(ab, ac)
#     d11 = jnp.dot(ac, ac)
#     d20 = jnp.dot(vp, ab)
#     d21 = jnp.dot(vp, ac)
#     denom = d00 * d11 - d01 * d01
#     denom = jnp.where(denom == 0, 1e-8, denom)

#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w

#     is_inside = (u >= 0) & (v >= 0) & (w >= 0)

#     # Closest point
#     v_clamped = jnp.clip(v, 0.0, 1.0)
#     w_clamped = jnp.clip(w, 0.0, 1.0)
#     u_clamped = 1.0 - v_clamped - w_clamped
#     closest_point = jnp.where(
#         is_inside,
#         proj_p,
#         u_clamped * v0 + v_clamped * v1 + w_clamped * v2
#     )

#     distance = jnp.linalg.norm(p - closest_point)
#     return distance

# @jax.jit
# def point_to_mesh_min_distance(p, v0, v1, v2):
#     # Function to compute minimum distance over triangles

#     def triangle_distance(carry, triangle):
#         min_dist = carry
#         v0_t, v1_t, v2_t = triangle
#         dist = point_triangle_distance(p, v0_t, v1_t, v2_t)
#         min_dist = jnp.minimum(min_dist, dist)
#         return min_dist, None

#     init_min_dist = jnp.inf
#     triangles = (v0, v1, v2)

#     min_distance, _ = jax.lax.scan(triangle_distance, init_min_dist, triangles, length=v0.shape[0])
#     return min_distance

# def compute_min_distances(points, mesh_vertices, mesh_faces, point_batch_size=1024, triangle_batch_size=1024):
#     num_points = points.shape[0]
#     num_triangles = mesh_faces.shape[0]
#     distances = []

#     for i in range(0, num_points, point_batch_size):
#         batch_points = points[i:i+point_batch_size]
#         batch_distances = []

#         for j in range(0, num_triangles, triangle_batch_size):
#             v0 = mesh_vertices[mesh_faces[j:j+triangle_batch_size, 0]]
#             v1 = mesh_vertices[mesh_faces[j:j+triangle_batch_size, 1]]
#             v2 = mesh_vertices[mesh_faces[j:j+triangle_batch_size, 2]]

#             # Vectorized over points in the batch
#             def compute_batch(p):
#                 return point_to_mesh_min_distance(p, v0, v1, v2)

#             batch_min_distances = jax.vmap(compute_batch)(batch_points)
#             batch_distances.append(batch_min_distances)

#         # Stack and minimize over triangle batches
#         batch_distances = jnp.stack(batch_distances, axis=1)
#         min_distances = jnp.min(batch_distances, axis=1)
#         distances.append(min_distances)

#     distances = jnp.concatenate(distances)
#     return distances

# def maximum_interpenetration_distance(mesh1, mesh2, point_batch_size=1024, triangle_batch_size=1024):
#     # Get inside masks
#     inside_mask1 = get_inside_mask(mesh1.vertices, mesh2.vertices, mesh2.faces)
#     inside_mask2 = get_inside_mask(mesh2.vertices, mesh1.vertices, mesh1.faces)

#     # Compute distances with batched processing
#     distances1 = compute_min_distances(
#         mesh1.vertices, mesh2.vertices, mesh2.faces,
#         point_batch_size, triangle_batch_size
#     )
#     distances2 = compute_min_distances(
#         mesh2.vertices, mesh1.vertices, mesh1.faces,
#         point_batch_size, triangle_batch_size
#     )

#     # Mask distances for points outside
#     distances1 = jnp.where(inside_mask1, distances1, -jnp.inf)
#     distances2 = jnp.where(inside_mask2, distances2, -jnp.inf)

#     # Compute maximum interpenetration distance
#     max_distance1 = jnp.max(distances1)
#     max_distance2 = jnp.max(distances2)
#     max_interpenetration_distance = jnp.maximum(max_distance1, max_distance2)
#     return max_interpenetration_distance


# def get_interpenetration(mesh_seq, num_sample):
#     interpenetrations = []
#     for pair in list(itertools.combinations(mesh_seq, 2)):
#         m1, m2 = pair
#         # Compute intersection distance
#         intersection_dist = maximum_interpenetration_distance(
#             m1, m2
#         )
#         interpenetrations.append(intersection_dist)
#     return jnp.array(interpenetrations).sum()


@jax.jit
def sample_uniform_broadcasted(key, low, high):
    return genjax.uniform.sample(key, low, high)


def logpdf_uniform_broadcasted(values, low, high):
    valid = (low <= values) & (values <= high)
    position_score = jnp.log((valid * 1.0) * (jnp.ones_like(values) / (high - low)))
    return position_score.sum()


uniform_broadcasted = genjax.exact_density(
    sample_uniform_broadcasted, logpdf_uniform_broadcasted
)


uniform_discrete = genjax.exact_density(
    lambda key, vals: jax.random.choice(key, vals),
    lambda sampled_val, vals: jnp.log(1.0 / (vals.shape[0])),
)
uniform_pose = genjax.exact_density(sample_uniform_pose, logpdf_uniform_pose)
uniform_scale = genjax.exact_density(sample_uniform_scale, logpdf_uniform_scale)

vmf = genjax.exact_density(
    lambda key, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).sample(seed=key),
    lambda x, mean, concentration: tfp.distributions.VonMisesFisher(
        mean, concentration
    ).log_prob(x),
)

gaussian_vmf = genjax.exact_density(sample_gaussian_vmf_pose, logpdf_gaussian_vmf_pose)

### Below are placeholders for genjax functions which are currently buggy ###

# There is currently a bug in `genjax.uniform.logpdf`; this `uniform`
# can be used instead until a fix is pushed.
uniform = genjax.exact_density(
    lambda key, low, high: genjax.uniform.sample(key, low, high),
    lambda x, low, high: jnp.sum(genjax.uniform.logpdf(x, low, high)),
)


def tfp_distribution(dist):
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return jnp.sum(d.log_prob(v))

    return genjax.exact_density(sampler, logpdf)


categorical = tfp_distribution(
    lambda logits: tfp.distributions.Categorical(logits=logits)
)
bernoulli = tfp_distribution(lambda logits: tfp.distributions.Bernoulli(logits=logits))
normal = tfp_distribution(tfp.distributions.Normal)
