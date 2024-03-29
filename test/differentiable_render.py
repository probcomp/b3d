import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
import rerun as rr
import functools

rr.init("demo22.py")
rr.connect("127.0.0.1:8812")

image_width=100
image_height=100
fx=50.0
fy=50.0
cx=50.0
cy=50.0
near=0.001
far=16.0
renderer = b3d.Renderer(
    image_width, image_height, fx, fy, cx, cy, near, far
)

## Render color
from pathlib import Path
mesh_path = Path(b3d.__file__).parents[1] / "assets/006_mustard_bottle/textured_simple.obj"
mesh = trimesh.load(mesh_path)

vertices = jnp.array(mesh.vertices) * 20.0
vertices = vertices - vertices.mean(0)
faces = jnp.array(mesh.faces)
vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0
ranges = jnp.array([[0, len(faces)]])

vertices = jnp.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])
faces = jnp.array([
    [0, 1, 2]
])
ranges = jnp.array([[0, 1]])
vertex_colors = jnp.array([
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
])

gt_pose = Pose.from_translation(
    jnp.array([0.0, 0.0, 3.0]),
)

target_image, depth = renderer.render_attribute(gt_pose.as_matrix()[None,...], vertices, faces, ranges, vertex_colors)
rr.log("/rgb", rr.Image(target_image), timeless=True)


@functools.partial(
    jnp.vectorize,
    signature="(3)->()",
    excluded=(
        1,
    ),
)
def point_to_line_distance(point, vector):
    unit_vector = vector / jnp.linalg.norm(vector)
    return jnp.linalg.norm(point - jnp.dot(point, unit_vector) * unit_vector)

width = 5
def get_mixed_color(ij, ijs, triangle_ids_padded, vertices_transformed_by_pose, faces, vertex_colors):
    i,j = ij
    triangle_ids_patch = jax.lax.dynamic_slice(
        triangle_ids_padded,
        jnp.array([i,j]),
        (2*width+1, 2*width+1),
    )
    ijs_patch = jax.lax.dynamic_slice(
        ijs,
        jnp.array([i,j, 0]),
        (2*width+1, 2*width+1,2),
    )
    
    pixel_vectors = jnp.concatenate([(ijs_patch - jnp.array([cx, cy])) / jnp.array([fx, fy]), jnp.ones_like(ijs_patch[...,0:1])], axis=-1)

    pixel_vector = pixel_vectors[width, width]

    center_points_of_triangle =(
        vertices_transformed_by_pose[faces[triangle_ids_patch-1]]
    ).mean(-2) * (triangle_ids_patch > 0)[...,None]

    distances = point_to_line_distance(center_points_of_triangle, pixel_vector)
    return distances.sum()
    # weights = 1 / distances
    # normalized_weights = weights / weights.sum()
    
    # colors_patch = (vertex_colors[faces[triangle_ids_patch-1]] * (triangle_ids_patch > 0)[...,None,None]).mean(-2)
    # final_color = (colors_patch * normalized_weights[...,None]).sum(0).sum(0)
    # return jnp.clip(final_color, 0.0, 1.0)


def f(pose):
    _, _, triangle_ids, _ = renderer.render(pose.as_matrix()[None,...], vertices, faces, ranges)
    vertices_transformed_by_pose = pose.apply(vertices)
    triangle_ids_padded = jnp.pad(triangle_ids, pad_width=[(width, width)])
    ijs = jnp.moveaxis(jnp.mgrid[: image_height, : image_width], 0, -1)
    mixed_image = get_mixed_color(jnp.array([55,55]), ijs, triangle_ids_padded, vertices_transformed_by_pose, faces, vertex_colors)
    return mixed_image.sum()

pose = Pose.from_translation(
    jnp.array([0.3, 0.3, 3.0]),
)
pose2 = Pose.from_translation(
    jnp.array([0.2, 0.2, 3.0]),
)
print(loss(pose), loss(pose2))

grad_func = jax.grad(loss)
print(grad_func(pose))



def loss(pose):
    return jnp.abs(render_mixed(pose) - target_image).sum()



def render_mixed(pose):
    _, _, triangle_ids, _ = renderer.render(pose.as_matrix()[None,...], vertices, faces, ranges)
    vertices_transformed_by_pose = pose.apply(vertices)

    triangle_ids_padded = jnp.pad(triangle_ids, pad_width=[(width, width)])
    ijs = jnp.moveaxis(jnp.mgrid[: image_height, : image_width], 0, -1)
    mixed_image = get_mixed_color(ijs, ijs, triangle_ids_padded, vertices_transformed_by_pose, faces, vertex_colors)
    return mixed_image

def loss(pose):
    return jnp.abs(render_mixed(pose) - target_image).sum()

def point_to_line_distance(point, vector):
    return jnp.linalg.norm(point - jnp.dot(point, unit_vector) * unit_vector)

def loss(point):
    return point_to_line_distance(point, jnp.ones(3) * 4.0)
jax.grad(loss)(jnp.ones(3))



mixed_image = render_mixed(pose)
rr.log("/rgb", rr.Image(target_image), timeless=True)
rr.log("/rgb/1", rr.Image(mixed_image), timeless=True)

grad_func = jax.grad(loss)
print(grad_func(pose))

for _ in range(100):
    pose_grad = grad_func(pose)
    pose = Pose(
        pose.pos - pose_grad.pos * 1000.0,
        pose.quat
    )
    print(loss(pose))
    mixed_image = render_mixed(pose)
    rr.log("/rgb", rr.Image(target_image), timeless=True)
    rr.log("/rgb/1", rr.Image(mixed_image), timeless=True)

