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

pose = Pose.from_position_and_target(
    jnp.array([3.2, 0.5, 0.0]),
    jnp.array([0.0, 0.0, 0.0])

).inverse()




@functools.partial(
    jnp.vectorize,
    signature="(3)->()",
    excluded=(
        1,
    ),
)
def point_to_line_distance(point, vector):
    return jnp.linalg.norm(jnp.cross(point, point - vector)) / jnp.linalg.norm(vector)

width = 30

gt_pose = Pose.from_position_and_target(
    jnp.array([3.2, 0.5, 0.0]),
    jnp.array([0.0, 0.0, 0.0])

).inverse()

target_image, depth = renderer.render_attribute(gt_pose.as_matrix()[None,...], vertices, faces, ranges, vertex_colors)


@functools.partial(
    jnp.vectorize,
    signature="(2)->(3)",
    excluded=(1,2,3,4,)
)
def get_mixed_color(ij, triangle_ids_padded, vertices_transformed_by_pose, faces, vertex_colors):
    i,j = ij
    triangle_ids_patch = jax.lax.dynamic_slice(
        triangle_ids_padded,
        jnp.array([ij[0], ij[1]]),
        (width, width),
    )
    center_points_of_triangle = vertices_transformed_by_pose[faces[triangle_ids_patch-1]].mean(-2)
    pixel_vector = jnp.array([(j - cx) / fx , (i - cy) / fy, 1.0])
    distances = 1.0 / point_to_line_distance(center_points_of_triangle, pixel_vector)**2
    colors_patch = vertex_colors[faces[triangle_ids_patch-1]].mean(-2)
    weights = distances / jnp.sum(distances)
    final_color = (colors_patch * weights[...,None]).sum(0).sum(0)
    return final_color * (triangle_ids_patch[width,width] > 0)


def render_mixed(pose):
    _, _, triangle_ids, _ = renderer.render(pose.as_matrix()[None,...], vertices, faces, ranges)
    vertices_transformed_by_pose = pose.apply(vertices)

    triangle_ids_padded = jnp.pad(triangle_ids, pad_width=[(width, width)])
    ijs = jnp.moveaxis(jnp.mgrid[: image_height, : image_width], 0, -1)
    mixed_image = get_mixed_color(ijs, triangle_ids_padded, vertices_transformed_by_pose, faces, vertex_colors)
    return mixed_image

def loss(pose):
    return jnp.abs(render_mixed(pose) - target_image).sum()



pose = Pose.from_position_and_target(
    jnp.array([3.0, 0.4, 0.0]),
    jnp.array([0.0, 0.0, 0.0])

).inverse()
mixed_image = render_mixed(pose)
rr.log("/rgb", rr.Image(target_image), timeless=True)
rr.log("/rgb/1", rr.Image(mixed_image), timeless=True)

grad_func = jax.grad(loss)

for _ in range(10):
    pose_grad = grad_func(pose)
    print(pose_grad)

    pose = pose - pose_grad * 0.0001
    print(loss(pose))
    mixed_image = render_mixed(pose)
    rr.log("/rgb", rr.Image(target_image), timeless=True)
    rr.log("/rgb/1", rr.Image(mixed_image), timeless=True)





colors_from_triangles = vertex_colors[faces[triangle_ids_patch-1][...,0]]
rr.log("color_in_filter", rr.Image(colors_from_triangles), timeless=True)


rr.log("color_in_filter", rr.Image(jnp.ones((2*width+1, 2*width+1, 3))
 * vertex_colors[faces[triangle_ids[i,j]][0]]), timeless=True)


rr.log("color_in_filter", rr.Image(jnp.ones((2*width+1, 2*width+1, 3)) * image[i,j] ), timeless=True)

