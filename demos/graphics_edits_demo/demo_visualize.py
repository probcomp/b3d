#!/usr/bin/env python
import os
import pickle

import b3d
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import trimesh
from b3d import Pose
from tqdm import tqdm

rr.init("demo_visualize3")
rr.connect("127.0.0.1:8812")

# Load date
# Load date
path = os.path.join(
    b3d.get_assets_path(),
    #  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
    # "shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
    "shared_data_bucket/input_data/desk_ramen2_spray1.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)


<<<<<<< HEAD
import pickle

=======
>>>>>>> main
data, object_library = pickle.load(open("demo_data.dat", "rb"))


# Get intrinsics
image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_depth
)
image_width, image_height = int(image_width), int(image_height)
fx, fy, cx, cy, near, far = (
    float(fx),
    float(fy),
    float(cx),
    float(cy),
    float(near),
    float(far),
)

# Get RGBS and Depth
rgbs = video_input.rgb[::3] / 255.0
xyzs = video_input.xyz[::3]

# Resize rgbs to be same size as depth.
rgbs_resized = jnp.clip(
    jax.vmap(jax.image.resize, in_axes=(0, None, None))(
        rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
    ),
    0.0,
    1.0,
)

renderer = b3d.RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)

object_positions_over_time = jnp.array(
    [
        jnp.zeros((4, 3)).at[: len(data[i][0])].set(data[i][0].pos)
        for i in range(len(data))
    ]
)
object_quaternions_over_time = jnp.array(
    [
        jnp.tile(b3d.Pose.identity_quaternion, (4, 1))
        .at[: len(data[i][0])]
        .set(data[i][0].quat)
        for i in range(len(data))
    ]
)
object_poses_over_time = Pose(object_positions_over_time, object_quaternions_over_time)
num_objects_in_frame_over_time = jnp.array([len(data[i][0]) for i in range(len(data))])

camera_poses_over_time = Pose(
    jnp.array([data[i][1].pos for i in range(len(data))]),
    jnp.array([data[i][1].quat for i in range(len(data))]),
)

transformed_vertices = (
    object_poses_over_time[0][object_library.vertex_index_to_object].apply(
        object_library.vertices
    )
    * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[..., None]
    )[..., None]
)

vertices = object_library.vertices


def compute_face_normals(vertices, faces):
    # Step 1: Calculate the face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Vectors for the sides of the triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Cross product to get face normals
    face_normals = jnp.cross(edge1, edge2)

    # Normalize the face normals
    face_normals = face_normals / jnp.linalg.norm(face_normals, axis=1, keepdims=True)
    return face_normals


def compute_vertex_normals(vertices, faces):
    face_normals = compute_face_normals(vertices, faces)

    # Step 2: Accumulate face normals to vertex normals
    vertex_normals = jnp.zeros_like(vertices)
    for i in range(3):
        vertex_normals = vertex_normals.at[faces[:, i]].add(face_normals)

    # Normalize the vertex normals
    vertex_normals = vertex_normals / jnp.linalg.norm(
        vertex_normals, axis=1, keepdims=True
    )

    return vertex_normals


def adjust_vertex_colors(
    vertices, faces, vertex_colors, light_position, ambient_light=0.1
):
    normals = compute_vertex_normals(vertices, faces)

    # Vector from vertices to light source
    light_vectors = light_position - vertices
    light_vectors = light_vectors / jnp.linalg.norm(
        light_vectors, axis=1, keepdims=True
    )

    # Dot product of normals and light vectors (cosine of angle)
    light_intensity = jnp.sum(normals * light_vectors, axis=1, keepdims=True)

    # Clamp values to range [0, 1]
    light_intensity = jnp.clip(light_intensity, 0.0, 1.0)

    # Combine ambient and diffuse lighting
    light_intensity = ambient_light + (1 - ambient_light) * light_intensity

    # Adjust vertex colors based on light intensity
    adjusted_colors = vertex_colors * light_intensity

    # Ensure colors are in range [0, 1]
    adjusted_colors = jnp.clip(adjusted_colors, 0.0, 1.0)

    return adjusted_colors


def viz_timestep(timestep, frame_number, vertices, faces, attributes, image=None):
    if image is None:
        rendered_rgbd = renderer.render_rgbd(vertices, faces, attributes)
    else:
        rendered_rgbd = image
    rr.set_time_sequence("frame", frame_number)
    rr.log("/rerenderings", rr.Image(rendered_rgbd[..., :3]))
    rr.log("/actual", rr.Image(rgbs_resized[timestep][..., :3]))


#### INITIAL PLAY
timestep = 0
frame_number = 0

rerenderings = []
for i in tqdm(range(100)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1


# MOVE OBJECT AROUND TABLE
plane_pose = b3d.Pose.fit_plane(object_library.objects[0][0], 0.01, 1000, 10000)
waypoints = [
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([-0.0, -0.1, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.1, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
]
interpolated_waypoints = jnp.concatenate(
    [
        jnp.linspace(waypoints[i], waypoints[i + 1], 15)
        for i in range(len(waypoints) - 1)
    ],
    axis=0,
)
for i in range(len(interpolated_waypoints)):
    object_poses = data[timestep][0].copy()
    camera_pose = data[timestep][1]
    new_pose = (
        plane_pose
        @ b3d.Pose.from_translation(interpolated_waypoints[i])
        @ plane_pose.inv()
        @ object_poses[1]
    )
    object_poses._position = object_poses.pos.at[1].set(new_pose.pos)
    object_poses._quaternion = object_poses.quat.at[1].set(new_pose.quat)

    object_poses_in_camera_frame = camera_pose.inv() @ object_poses

    attributes = jnp.array(object_library.attributes)

    transformed_vertices = (
        object_poses_in_camera_frame[object_library.vertex_index_to_object](
            object_library.vertices
        )
        * (object_library.vertex_index_to_object < len(object_poses))[..., None]
    )
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number = frame_number + 1


#### REPLACE the object with a blue texture
for i in tqdm(range(60)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep][object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        (
            object_library.vertex_index_to_object
            < (num_objects_in_frame_over_time[timestep, None])
        )[..., None]
    )

    new_colors = adjust_vertex_colors(
        transformed_vertices,
        object_library.faces,
        object_library.attributes * 0.0 + jnp.array([0.0, 0.0, 0.8]),
        jnp.array([0.3, 0.3, 0.2]),
        ambient_light=0.5,
    )
    mask = (object_library.vertex_index_to_object == 1)[..., None]

    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        new_colors * mask + object_library.attributes * ~mask,
    )
    frame_number += 1
    timestep += 1

#### PLAY MORE FRAMES
for i in tqdm(range(10)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1

#### FADE OUT THE BACKGROUND
waypoints = [jnp.array([1.0]), jnp.array([0.5]), jnp.array([1.0])]
interpolated_waypoints = jnp.concatenate(
    [
        jnp.linspace(waypoints[i], waypoints[i + 1], 20)
        for i in range(len(waypoints) - 1)
    ],
    axis=0,
)
for i in range(len(interpolated_waypoints)):
    object_poses = data[timestep][0].copy()
    camera_pose = data[timestep][1]
    object_poses_in_camera_frame = camera_pose.inv() @ object_poses

    transformed_vertices = (
        object_poses_in_camera_frame[object_library.vertex_index_to_object](
            object_library.vertices
        )
        * (object_library.vertex_index_to_object < len(object_poses))[..., None]
    )
    mask = (object_library.vertex_index_to_object == 0)[..., None]
    attributes = object_library.attributes
    attributes_ = (
        jnp.clip(attributes * interpolated_waypoints[i, 0], 0.0, 1.0) * mask
        + attributes * ~mask
    )

    viz_timestep(
        timestep, frame_number, transformed_vertices, object_library.faces, attributes_
    )
    frame_number += 1
    timestep += 1


#### PLAY MORE FRAMES
for i in tqdm(range(20)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1


print(frame_number)
### Replace Objects with Box
rr.set_time_sequence("frame", frame_number)
object_vertices = object_library.objects[2][0]
object_pose = object_poses_over_time[timestep, 2]
object_vertices_in_world_frame = (object_pose).apply(object_vertices)

rr.log("background", rr.Points3D(object_vertices_in_world_frame))
rr.log("object_vertices", rr.Points3D(object_vertices))

b3d.rr_log_pose("/object_poses/box", camera_pose @ object_pose)

aabb_dims, aabb_pose = b3d.aabb(object_vertices_in_world_frame)
box_mesh = trimesh.load(os.path.join(b3d.get_assets_path(), "objs/cube.obj"))
box_vertices, box_faces = jnp.array(box_mesh.vertices), jnp.array(box_mesh.faces)
box_vertices = box_vertices * jnp.array([0.08, 0.1, 0.08])[None]
box_colors = jnp.tile(jnp.array([0.05, 0.6, 0.05]), (len(box_vertices), 1))
box_vertices = object_pose.inv().apply(aabb_pose.apply(box_vertices))
rr.log("box_vertices", rr.Points3D(box_vertices))

object_library.add_object(box_vertices, box_faces, box_colors)
num_objects = jnp.max(object_library.vertex_index_to_object)


for i in tqdm(range(80)):
    concatenated_object_poses = b3d.Pose.concatenate_poses(
        [
            object_poses_over_time[timestep],
            object_poses_over_time[timestep][2][None, ...],
        ]
    )
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ concatenated_object_poses[object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        jnp.logical_or(
            object_library.vertex_index_to_object
            < (num_objects_in_frame_over_time[timestep, None] - 1),
            object_library.vertex_index_to_object == num_objects,
        )
    )[..., None]

    new_colors = adjust_vertex_colors(
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
        jnp.array([0.0, 0.0, 0.0]),
        ambient_light=0.8,
    )
    mask = (object_library.vertex_index_to_object == num_objects)[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        new_colors * mask + object_library.attributes * ~mask,
    )
    frame_number += 1
    timestep += 1
print(frame_number)


#### PLAY MORE FRAMES
for i in tqdm(range(50)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1


# saved_frame_number = frame_number
# saved_timestep = timestep

# frame_number = saved_frame_number
# timestep = saved_timestep

# MOVE OBJECT AROUND TABLE
waypoints = [
    jnp.array([0.0]),
    jnp.array([-0.8]),
    jnp.array([0.0]),
    jnp.array([0.8]),
    jnp.array([0.0]),
]
interpolated_waypoints = jnp.concatenate(
    [
        jnp.linspace(waypoints[i], waypoints[i + 1], 10)
        for i in range(len(waypoints) - 1)
    ],
    axis=0,
)
for i in range(len(interpolated_waypoints)):
    object_poses = data[timestep][0].copy()
    camera_pose = data[timestep][1]
    plane_rotation = b3d.Pose.from_quat(plane_pose.quat)
    rotation = b3d.Pose.from_quat(
        b3d.Rot.from_rotvec(
            jnp.array([0.0, 0.0, interpolated_waypoints[i, 0]])
        ).as_quat()
    )
    new_pose = object_poses[3] @ plane_rotation @ rotation @ plane_rotation.inv()
    object_poses._position = object_poses.pos.at[3].set(new_pose.pos)
    object_poses._quaternion = object_poses.quat.at[3].set(new_pose.quat)
    object_poses_in_camera_frame = camera_pose.inv() @ object_poses

    attributes = jnp.array(object_library.attributes)

    transformed_vertices = (
        object_poses_in_camera_frame[object_library.vertex_index_to_object](
            object_library.vertices
        )
        * (object_library.vertex_index_to_object < len(object_poses))[..., None]
    )
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number = frame_number + 1


#### PLAY MORE FRAMES
for i in tqdm(range(10)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1


# frame_number = saved_frame_number
# timestep = saved_timestep

#### CHANGE COLOR OF SPRAY BOTTLE
for i in range(40):
    object_poses = data[timestep][0].copy()
    camera_pose = data[timestep][1]
    object_poses_in_camera_frame = camera_pose.inv() @ object_poses
    transformed_vertices = (
        object_poses_in_camera_frame[object_library.vertex_index_to_object](
            object_library.vertices
        )
        * (object_library.vertex_index_to_object < len(object_poses))[..., None]
    )
    mask = (object_library.vertex_index_to_object == 3)[..., None]
    attributes = object_library.attributes
    attributes_ = attributes[:, jnp.array([1, 0, 2])] * mask + attributes * ~mask

    viz_timestep(
        timestep, frame_number, transformed_vertices, object_library.faces, attributes_
    )
    frame_number += 1
    timestep += 1

#### PLAY MORE FRAMES
for i in tqdm(range(20)):
    transformed_vertices = (
        camera_poses_over_time[timestep].inv()
        @ object_poses_over_time[timestep, object_library.vertex_index_to_object]
    ).apply(object_library.vertices) * (
        object_library.vertex_index_to_object
        < num_objects_in_frame_over_time[timestep, None]
    )[..., None]
    viz_timestep(
        timestep,
        frame_number,
        transformed_vertices,
        object_library.faces,
        object_library.attributes,
    )
    frame_number += 1
    timestep += 1


# for i in tqdm(range(60)):
#     concatenated_object_poses = b3d.Pose.concatenate_poses([object_poses_over_time[timestep], object_poses_over_time[timestep][2][None,...]])
#     transformed_vertices = (camera_poses_over_time[timestep].inv() @ concatenated_object_poses[object_library.vertex_index_to_object]).apply(
#         object_library.vertices) * (
#         jnp.logical_or(
#             object_library.vertex_index_to_object < (num_objects_in_frame_over_time[timestep,None] - 1),
#             object_library.vertex_index_to_object == num_objects))[...,None]

#     face_normals = compute_face_normals(transformed_vertices, object_library.faces)
#     face_centers = transformed_vertices[object_library.faces].mean(-1)

#     rendered_rgbd = renderer.render_rgbd(
#         transformed_vertices,
#         object_library.faces,
#         object_library.attributes
#     )
#     rasterize_out = renderer.rasterize(
#         transformed_vertices,
#         object_library.faces,
#     )
#     face_centers_image = face_centers[rasterize_out[...,-1].astype(jnp.int32) - 1]
#     normals_image = face_normals[rasterize_out[...,-1].astype(jnp.int32) -  1]

#     light_position = jnp.array([0.0, 0.0, 0.0])

#     # Vector from vertices to light source
#     light_vectors = light_position - face_centers_image
#     light_vectors = light_vectors / jnp.linalg.norm(light_vectors, axis=-1, keepdims=True)

#     light_intensity = jnp.sum(normals_image * light_vectors, axis=-1, keepdims=True)

#     # Clamp values to range [0, 1]
#     light_intensity = jnp.clip(light_intensity, 0.0, 1.0)

#     # Combine ambient and diffuse lighting
#     ambient_light = 1.5
#     light_intensity = ambient_light + (1 - ambient_light) * light_intensity

#     # Adjust vertex colors based on light intensity
#     adjusted_colors = rendered_rgbd * light_intensity

#     # Ensure colors are in range [0, 1]
#     adjusted_colors = jnp.clip(adjusted_colors, 0.0, 1.0)

#     mask =

#     viz_timestep(timestep, frame_number, transformed_vertices, object_library.faces, None,
#                     image = adjusted_colors[...,:3] * mask + rendered_rgbd[...,:3] * ~mask
#         )
#     frame_number += 1
#     timestep += 1
