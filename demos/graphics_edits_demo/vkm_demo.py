import jax.numpy as jnp
import os
import b3d
import rerun as rr
import jax
import argparse
import numpy as np
from b3d import Pose

rr.init("vkm_demo2")
rr.connect("127.0.0.1:8812")

filename = os.path.join(
    b3d.get_assets_path(),
    #  "shared_data_bucket/input_data/orange_mug_pan_around_and_pickup.r3d.video_input.npz")
    # "shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz")
    "shared_data_bucket/input_data/409_bottle.r3d.video_input.npz.downsampled.npz",
)
video_input = b3d.io.VideoInput.load(filename)

# video_input = b3d.VideoInput(
#     rgb=video_input.rgb[::5],
#     xyz=video_input.xyz[::5],
#     camera_positions=video_input.camera_positions[::5],
#     camera_quaternions=video_input.camera_quaternions[::5],
#     camera_intrinsics_rgb=video_input.camera_intrinsics_rgb,
#     camera_intrinsics_depth=video_input.camera_intrinsics_depth,
# )
# video_input.save(filename + ".downsampled")

image_width, image_height, fx, fy, cx, cy, near, far = np.array(
    video_input.camera_intrinsics_rgb
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


camera_poses_over_time = Pose(
    video_input.camera_positions, video_input.camera_quaternions
)

def _resize_rgb_or_xyz(d):
    return jax.image.resize(d, (image_height, image_width, 3), "linear")
resize_rgb_or_xyz = jnp.vectorize(_resize_rgb_or_xyz, signature='(h,w,3)->(a,b,3)')

# Take point cloud at frame 0
point_cloud = camera_poses_over_time[0].apply(jax.image.resize(
    video_input.xyz[0], (image_height, image_width, 3), "linear"
).reshape(-1, 3))
point_cloud_colors = jax.image.resize(
    video_input.rgb[0] / 255.0, (image_height, image_width, 3), "linear"
).reshape(-1, 3)
assert point_cloud.shape == point_cloud_colors.shape

# `make_mesh_from_point_cloud_and_resolution` takes a 3D positions, colors, and sizes of the boxes that we want
# to place at each position and create a mesh
background_vertices, background_faces, background_vertex_colors, face_colors = (
    b3d.make_mesh_from_point_cloud_and_resolution(
        point_cloud,
        point_cloud_colors,
        point_cloud[:, 2]
        / fx
        * 6.0,  # This is scaling the size of the box to correspond to the effective size of the pixel in 3D. It really should be multiplied by 2.
        # and the 6 makes it larger
    )
)

renderer = b3d.RendererOriginal(image_width, image_height, fx, fy, cx, cy, near, far)

start_t, end_t = 36, 100


visualization_images = []

for t in range(start_t):
    rr.set_time_sequence("frame", t)
    rgb_rerender = renderer.render(camera_poses_over_time[t].inv().apply(background_vertices), background_faces, background_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))

voxel_occupied_occluded_free_jit = jax.jit(b3d.voxel_occupied_occluded_free)
voxel_occupied_occluded_free_parallel_camera = jax.jit(
    jax.vmap(b3d.voxel_occupied_occluded_free, in_axes=(0, None, None, None, None, None, None, None, None, None))
)
voxel_occupied_occluded_free_parallel_camera_depth = jax.jit(
    jax.vmap(b3d.voxel_occupied_occluded_free, in_axes=(0, 0, 0, None, None, None, None, None, None, None))
)

# OBJECT ACQUISIOTION

camera_poses_for_acquisition = camera_poses_over_time[start_t:end_t]
rgbs_for_acquisition = resize_rgb_or_xyz(video_input.rgb[start_t:end_t] / 255.0)
xyz_for_acquisition = resize_rgb_or_xyz(video_input.xyz[start_t:end_t])
masks_concat = jnp.stack([b3d.carvekit_get_foreground_mask(r) for r in rgbs_for_acquisition])

mask_0 = b3d.carvekit_get_foreground_mask(rgbs_for_acquisition[0])

point_cloud_at_start = camera_poses_over_time[start_t].apply(xyz_for_acquisition[0])[mask_0]
rr.log("cloud", rr.Points3D(point_cloud_at_start, colors=rgbs_for_acquisition[0][mask_0].reshape(-1,3)))

grid_center = jnp.median(point_cloud_at_start,axis=0)

W = 0.25
D = 150
grid = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-W/2, +W/2, D) * 0.8,
        jnp.linspace(-W/2, +W/2, D) * 1.2,
        jnp.linspace(-W/2, +W/2, D) * 0.8,
    ),
    axis=-1,
).reshape(-1, 3) + grid_center
rr.log("grid", rr.Points3D(grid))


def merge_objects(objects, poses=None):
    if poses:
        merged_vertices = jnp.concatenate([poses[i].apply(v) for (i,(v, _, _)) in enumerate(objects)], axis=0)
    else:
        merged_vertices = jnp.concatenate([v for v, _, _ in objects], axis=0)
    vertices_cumsum = jnp.cumsum(jnp.array([0] + [v.shape[0] for v, _, _ in objects]))
    merged_faces = jnp.concatenate([f + vertices_cumsum[i] for i, (_, f, _) in enumerate(objects)], axis=0)
    merged_vertex_colors = jnp.concatenate([c for _, _, c in objects], axis=0)

    vertex_to_index = jnp.concatenate([jnp.full(v.shape[0], i) for i, (v, _, _) in enumerate(objects)])
    return merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index

objects = [
    (background_vertices, background_faces, background_vertex_colors),
    (jnp.zeros((0,3)), jnp.zeros((0,3),dtype=jnp.int32), jnp.zeros((0,3)))
]

poses = [Pose.identity(), Pose.identity()]
merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses=poses)

for t in range(start_t, end_t):
    if t % 5 == 0:
        occ_free_occl_, colors_per_voxel_ = voxel_occupied_occluded_free_parallel_camera_depth(
            camera_poses_for_acquisition[:t-start_t], rgbs_for_acquisition[:t-start_t], xyz_for_acquisition[...,2][:t-start_t] * masks_concat[:t-start_t] + (1.0 - masks_concat[:t-start_t]) * 5.0, grid, fx,fy,cx,cy, 6.0, 0.0025
        )

        i = len(occ_free_occl_)
        occ_free_occl, colors_per_voxel = occ_free_occl_[:i], colors_per_voxel_[:i]
        total_occ = (occ_free_occl == 1.0).sum(0)
        total_free = (occ_free_occl == -1.0).sum(0)
        ratio = total_occ / (total_occ + total_free) * ((total_occ + total_free) > 1)


        grid_colors = colors_per_voxel.sum(0)/ (total_occ[...,None])
        model_mask = ratio > 0.2
        resolution = 0.0015
        vertices, faces, vertex_colors, face_colors = b3d.make_mesh_from_point_cloud_and_resolution(
            grid[model_mask], grid_colors[model_mask], resolution * jnp.ones_like(model_mask) * 2.0
        )
        object_pose =  Pose.from_translation(vertices.mean(0))
        vertices_centered = object_pose.inverse().apply(vertices)
        poses[1] = object_pose

        objects[1] = (vertices_centered, faces, vertex_colors)
        merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses)

    rr.set_time_sequence("frame", t)
    rgb_rerender = renderer.render(camera_poses_over_time[t].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))

objects = objects[:2]
poses = poses[:2]

objects.append(objects[-1])
objects.append(objects[-1])

poses.append(poses[-1] @ Pose.from_translation(jnp.array([-0.1, 0.0, 0.1])))
poses.append(poses[-1] @ Pose.from_translation(jnp.array([-0.1, 0.0, -0.1])))

# Duplicate object
t = end_t
merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses)

for t in range(end_t, len(video_input.rgb)):
    rr.set_time_sequence("frame", t)
    rgb_rerender = renderer.render(camera_poses_over_time[t].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))


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
    vertex_normals = vertex_normals / jnp.linalg.norm(vertex_normals, axis=1, keepdims=True)

    return vertex_normals



def adjust_vertex_colors(vertices, faces, vertex_colors, light_position, ambient_light=0.1):
    normals = compute_vertex_normals(vertices, faces)

    # Vector from vertices to light source
    light_vectors = light_position - vertices
    light_vectors = light_vectors / jnp.linalg.norm(light_vectors, axis=1, keepdims=True)

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

t = len(video_input.rgb)

rr.log("transformed_vertices", rr.Points3D(merged_vertices[::3]))

transformed_vertices = camera_poses_over_time[t].inv().apply(merged_vertices)

#### LIGHTING
waypoints = [jnp.array([0.0, 0.0, 0.0, 1.0]), jnp.array([-0.2, -0.1, 0.4, 0.5]), jnp.array([0.3, -0.1, 0.4, 0.5]), jnp.array([0.0, 0.1, 0.2, 1.0])]
interpolated_waypoints = jnp.concatenate([jnp.linspace(waypoints[i], waypoints[i+1], 20) for i in range(len(waypoints)-1)], axis=0)

for i in range(len(interpolated_waypoints)):
    new_vertex_colors = adjust_vertex_colors(transformed_vertices, merged_faces, merged_vertex_colors, interpolated_waypoints[i,:3], ambient_light=float(interpolated_waypoints[i,3]))
    rr.set_time_sequence("frame", t + i)
    rgb_rerender = renderer.render(transformed_vertices, merged_faces, new_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[t] / 255.0, (image_height, image_width, 3), "linear"
    )))


## CHANGE OBJECTS
old_objects = [[jnp.copy(a) for a in b] for b in objects]

frame_index = len(video_input.rgb) + len(interpolated_waypoints) - 1
ticker = 0

for t in range(15):
    rr.set_time_sequence("frame", frame_index)
    rgb_rerender = renderer.render(camera_poses_over_time[-1-ticker].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    frame_index += 1
    ticker += 1


import trimesh
mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/019_pitcher_base/textured.obj"
)
mesh = trimesh.load(mesh_path)
mesh_list = list(b3d.get_vertices_faces_colors_from_mesh(mesh))
objects[3] = mesh_list
objects[3][0] = objects[3][0][:, jnp.array([0, 2, 1])] * jnp.array([1.0, -1.0, 1.0]) - jnp.array([0.0, 0.02, 0.0])

merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses)


for t in range(15):
    rr.set_time_sequence("frame", frame_index)
    rgb_rerender = renderer.render(camera_poses_over_time[-1-ticker].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    frame_index += 1
    ticker += 1



mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/021_bleach_cleanser/textured.obj"
)
mesh = trimesh.load(mesh_path)
mesh_list = list(b3d.get_vertices_faces_colors_from_mesh(mesh))
objects[2] = mesh_list
objects[2][0] = objects[2][0][:, jnp.array([0, 2, 1])] * jnp.array([1.0, -1.0, 1.0])

merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses)

for t in range(15):
    rr.set_time_sequence("frame", frame_index)
    rgb_rerender = renderer.render(camera_poses_over_time[-1-ticker].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    frame_index += 1
    ticker += 1


mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
mesh_list = list(b3d.get_vertices_faces_colors_from_mesh(mesh))
objects[1] = mesh_list
objects[1][0] = objects[1][0][:, jnp.array([0, 2, 1])] * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 0.07, 0.0])

merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(objects, poses)

for t in range(60):
    rr.set_time_sequence("frame", frame_index)
    rgb_rerender = renderer.render(camera_poses_over_time[-1-ticker].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    rr.log("rgb/rerender", rr.Image(jnp.clip(rgb_rerender, 0.0, 1.0)))
    rr.log("rgb", rr.Image( jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    visualization_images.append((rgb_rerender, jax.image.resize(
        video_input.rgb[-1-ticker] / 255.0, (image_height, image_width, 3), "linear"
    )))
    frame_index += 1
    ticker += 1

saved_ticker = ticker
saved_frame_index = frame_index

camera_pose = camera_poses_over_time[-1-ticker + 1]

merged_vertices, merged_faces, merged_vertex_colors, vertex_to_index = merge_objects(old_objects, poses)
merged_vertices_no_background, merged_faces_no_background, merged_vertex_colors_no_background, vertex_to_index_no_background = merge_objects(old_objects[1:], poses[1:])

old_objects2 = [[jnp.copy(a) for a in b] for b in old_objects]
for (i, color) in [(1, jnp.array([0.6, 0.05, 0.05])), (2, jnp.array([0.05, 0.6, 0.05])), (3, jnp.array([0.05, 0.05, 0.6]))]:
    object_pose = poses[i]
    object_vertices_in_object_frame = old_objects2[i][0]
    aabb_dims, aabb_pose = b3d.aabb(object_vertices_in_object_frame)
    box_mesh = trimesh.load(os.path.join(b3d.get_assets_path(), "objs/cube.obj"))
    box_vertices, box_faces = jnp.array(box_mesh.vertices), jnp.array(box_mesh.faces)
    box_vertices_modified = box_vertices *  aabb_dims * 0.8
    box_colors = jnp.tile(color, (len(box_vertices), 1))
    old_objects2[i] = (box_vertices_modified, box_faces, box_colors)

merged_vertices2, merged_faces2, merged_vertex_colors2, vertex_to_index2 = merge_objects(old_objects2[1:], poses[1:])

points = merged_vertices[vertex_to_index > 0]
points = points[jax.random.choice(jax.random.PRNGKey(0), points.shape[0], (1000,), replace=False),:]

# vertex_colors_random =  jax.random.uniform(jax.random.PRNGKey(0), (len(merged_vertex_colors[vertex_to_index > 0]), 3))

random_subset = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(merged_faces_no_background)), (4000,), replace=False)

viz_images = []
for i in range(len(camera_poses_over_time)):

    camera_pose = camera_poses_over_time[i]

    # Original
    transformed_vertices = camera_pose.inv().apply(merged_vertices)
    rgb_rerender = renderer.render(transformed_vertices, merged_faces, merged_vertex_colors)
    rerender_viz = b3d.get_rgb_pil_image(rgb_rerender)

    # Triangle mesh representation
    transformed_vertices = camera_pose.inv().apply(merged_vertices_no_background)
    rgb_rerender = renderer.render(transformed_vertices, merged_faces_no_background[random_subset], merged_vertex_colors_no_background)
    triangle_mesh_viz = b3d.get_rgb_pil_image(rgb_rerender)
    # triangle_mesh_viz.save("1.png")

    # Point light representation
    pixel_coords = b3d.xyz_to_pixel_coordinates(
        camera_pose.inv().apply(points),  fx, fy, cx, cy
    )
    pixel_coords = pixel_coords[(pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < image_height) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < image_width)]
    img = jnp.zeros((image_height, image_width))
    xs,ys = jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)
    w = 1
    for i in range(-w, w+1):
        for j in range(-w, w+1):
            img = img.at[xs+i,ys+j].set(1.0)
    img = jnp.tile(1.0 * (img > 0)[...,None], (1,1,3))
    point_light_viz = b3d.get_rgb_pil_image(img)
    point_light_viz.save("0.png")



    # Bounding Box representation
    transformed_vertices = camera_pose.inv().apply(merged_vertices2)
    rgb_rerender = renderer.render(transformed_vertices, merged_faces2, merged_vertex_colors2)
    bounding_box_viz = b3d.get_rgb_pil_image(rgb_rerender)
    # bounding_box_viz.save("1.png")

    viz_images.append(b3d.multi_panel(
        [rerender_viz, triangle_mesh_viz, point_light_viz, bounding_box_viz],
        labels=["Latent Scene", "Triangles", "Point Light", "Bounding Box"],
        label_fontsize=60
    ))

b3d.make_video_from_pil_images(viz_images, "viz_images.mp4", fps=10.0)



rgb_rerender = renderer.render(camera_poses_over_time[-1-ticker].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
b3d.get_rgb_pil_image(rgb_rerender).save("0.png")









def _pixel_coordinates_to_image(pixel_coords, image_height, image_width):
    img = jnp.zeros((image_height, image_width))
    img = img.at[jnp.round(pixel_coords[:, 0]).astype(jnp.int32), jnp.round(pixel_coords[:, 1]).astype(jnp.int32)].set(jnp.arange(len(pixel_coords))+1 )
    return img




from jax.lax import cond

def slerp(q1, q2, t):
    dot_product = jnp.dot(q1, q2)

    q1 = cond(dot_product < 0.0, lambda q: -q, lambda q: q, q1)
    dot_product = jnp.abs(dot_product)

    theta = jnp.arccos(jnp.clip(dot_product, -1.0, 1.0))
    sin_theta = jnp.sin(theta)

    def small_angle_case(_):
        return q1  # If the angle is very small, return q1

    def normal_case(_):
        w1 = jnp.sin((1 - t) * theta) / sin_theta
        w2 = jnp.sin(t * theta) / sin_theta
        return w1 * q1 + w2 * q2

    return cond(theta < 1e-6, small_angle_case, normal_case, operand=None)

def lerp(T1, T2, t):
    return (1 - t) * T1 + t * T2

new_camera_pose = camera_pose @ Pose.from_translation(jnp.array([0.1, 0.05, 0.3])) @ Pose.from_quat(b3d.Rot.from_rotvec(jnp.array([0.0, -0.9, 0.0])).as_quat())
rgb_rerender = renderer.render(new_camera_pose.inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
b3d.get_rgb_pil_image(rgb_rerender).save("0.png")

interpolated_camera_poses = [
    Pose(
        lerp(camera_pose.pos, new_camera_pose.pos, t),
        slerp(camera_pose.quat, new_camera_pose.quat, t)
    )
    for t in jnp.linspace(0.0, 1.0, 40)
]

new_images = []
for i in range(len(interpolated_camera_poses)):
    rgb_rerender = renderer.render(interpolated_camera_poses[i].inv().apply(merged_vertices), merged_faces, merged_vertex_colors)
    new_images.append((jnp.clip(rgb_rerender, 0.0, 1.0), visualization_images[-1][1]))

rendered_pil_images = [b3d.get_rgb_pil_image(r) for r, _ in (new_images)]
b3d.make_video_from_pil_images(rendered_pil_images, "rendered.mp4", fps=10.0)


rendered_pil_images = [b3d.get_rgb_pil_image(r) for r, _ in (visualization_images +  new_images)]
observed_pil_images = [b3d.get_rgb_pil_image(r) for _, r in (visualization_images +  new_images)]

b3d.make_video_from_pil_images(rendered_pil_images, "rendered.mp4", fps=10.0)
b3d.make_video_from_pil_images(observed_pil_images, "observed.mp4", fps=10.0)
