import inspect
import os
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import TypeAlias

import cv2
import distinctipy
import genjax
import jax
import jax.numpy as jnp
import numpy as np
import rerun as rr
import sklearn.cluster
import trimesh
from jax.scipy.spatial.transform import Rotation as Rot
from PIL import Image, ImageDraw, ImageFont

import b3d
from b3d.pose import Pose

# TODO: Refactor utils into core and others, to avoid circular imports

# # # # # # # # # # # #
#
#  Core
#
# # # # # # # # # # # #


def get_root_path() -> Path:
    return Path(Path(b3d.__file__).parents[2])


def get_assets() -> Path:
    """The absolute path of the assets directory on current machine"""
    assets_dir_path = get_root_path() / "assets"

    if not os.path.exists(assets_dir_path):
        os.makedirs(assets_dir_path)
        print(
            f"Initialized empty directory for shared bucket data at {assets_dir_path}."
        )

    return assets_dir_path


get_assets_path = get_assets


def make_assets_path(subpath):
    return get_assets() / subpath


def get_shared() -> Path:
    """The absolute path of the assets directory on current machine"""
    data_dir_path = get_assets() / "shared_data_bucket"

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        print(f"Initialized empty directory for shared bucket data at {data_dir_path}.")

    return data_dir_path


def get_gcloud_bucket_ref() -> str:
    return "gs://b3d_bucket"


def keysplit(key, *ns):
    if len(ns) == 0:
        return jax.random.split(key, 1)[0]
    elif len(ns) == 1:
        (n,) = ns
        if n == 1:
            return keysplit(key)
        else:
            return jax.random.split(key, ns[0])
    else:
        keys = []
        for n in ns:
            keys.append(keysplit(key, n))
        return keys


@jax.jit
def split_key(key):
    return jax.random.split(key, 2)[-1]


# # # # # # # # # # # #
#
#  Other
#
# # # # # # # # # # # #


@partial(jax.jit, static_argnums=(1, 2))
def resize_image_nearest(rgbd, height, width):
    return jax.image.resize(rgbd, (height, width, *rgbd.shape[2:]), method="nearest")


resize_image_nearest_vmap = jax.jit(
    jax.vmap(resize_image_nearest, in_axes=(0, None, None)), static_argnums=(1, 2)
)


@partial(jax.jit, static_argnums=(1, 2))
def resize_image_linear(rgbd, height, width):
    return jax.image.resize(rgbd, (height, width, *rgbd.shape[2:]), method="linear")


resize_image_linear_vmap = jax.jit(
    jax.vmap(resize_image_linear, in_axes=(0, None, None)), static_argnums=(1, 2)
)


@jax.jit
def clip_rgb(rgb):
    return jnp.clip(rgb, 0.0, 1.0)


@partial(jax.jit, static_argnums=1)
def downsize_images(ims, k):
    """Downsize an array of images by a given factor."""
    shape = (ims.shape[1] // k, ims.shape[2] // k, ims.shape[3])
    return jax.vmap(jax.image.resize, (0, None, None))(ims, shape, "linear")


@jax.jit
def xyz_from_depth(z: rr.DepthImage, fx, fy, cx, cy):
    v, u = jnp.mgrid[: z.shape[0], : z.shape[1]]
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz


xyz_from_depth_vectorized = jnp.vectorize(
    xyz_from_depth,
    excluded=(
        1,
        2,
        3,
        4,
    ),
    signature="(h,w)->(h,w,3)",
)


def xyz_to_pixel_coordinates(xyz, fx, fy, cx, cy):
    x = fx * xyz[..., 0] / (xyz[..., 2]) + cx
    y = fy * xyz[..., 1] / (xyz[..., 2]) + cy
    return jnp.stack([y, x], axis=-1)


def segment_point_cloud(point_cloud, threshold=0.01, min_points_in_cluster=0):
    c = sklearn.cluster.DBSCAN(eps=threshold).fit(point_cloud)
    labels = c.labels_
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    counter = 0
    new_labels = np.array(labels)
    for index in order:
        if unique[index] == -1:
            continue
        if counts[index] >= min_points_in_cluster:
            val = counter
        else:
            val = -1
        new_labels[labels == unique[index]] = val
        counter += 1
    return new_labels


def aabb(object_points):
    """
    Returns the axis aligned bounding box of a point cloud.
    Args:
        object_points: (N,3) point cloud
    Returns:
        dims: (3,) dimensions of the bounding box
        pose: (4,4) pose of the bounding box
    """
    maxs = jnp.max(object_points, axis=0)
    mins = jnp.min(object_points, axis=0)
    dims = maxs - mins
    center = (maxs + mins) / 2
    return dims, b3d.Pose.from_translation(center)


def pad_with_1(x):
    return jnp.concatenate((x, jnp.ones((*x.shape[:-1], 1))), axis=-1)


def make_mesh_from_point_cloud_and_resolution(grid_centers, grid_colors, resolutions):
    box_mesh = trimesh.load(os.path.join(b3d.get_assets_path(), "objs/cube.obj"))
    base_vertices, base_faces = jnp.array(box_mesh.vertices), jnp.array(box_mesh.faces)

    def process_ith_ball(i, positions, colors, base_vertices, base_faces, resolutions):
        transform = Pose.from_translation(positions[i])
        new_vertices = base_vertices * resolutions[i]
        new_vertices = transform.apply(new_vertices)
        return (
            new_vertices,
            base_faces + i * len(new_vertices),
            jnp.tile(colors[i][None, ...], (len(base_vertices), 1)),
            jnp.tile(colors[i][None, ...], (len(base_faces), 1)),
        )

    vertices_, faces_, vertex_colors_, face_colors_ = jax.vmap(
        process_ith_ball, in_axes=(0, None, None, None, None, None)
    )(
        jnp.arange(len(grid_centers)),
        grid_centers,
        grid_colors,
        base_vertices,
        base_faces,
        resolutions * 1.0,
    )

    vertices = jnp.concatenate(vertices_, axis=0)
    faces = jnp.concatenate(faces_, axis=0)
    vertex_colors = jnp.concatenate(vertex_colors_, axis=0)
    face_colors = jnp.concatenate(face_colors_, axis=0)
    return vertices, faces, vertex_colors, face_colors


def get_vertices_faces_colors_from_mesh(mesh):
    vertices = jnp.array(mesh.vertices)
    vertices = vertices - jnp.mean(vertices, axis=0)
    faces = jnp.array(mesh.faces)
    vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
    return vertices, faces, vertex_colors


def get_rgb_pil_image(image, max=1.0):
    """Convert an RGB image to a PIL image.

    Args:
        image (np.ndarray): RGB image. Shape (H, W, 3).
        max (float): Maximum value for colormap.
    Returns:
        PIL.Image: RGB image visualized as a PIL image.
    """
    image = np.clip(image, 0.0, max)
    # if image.shape[-1] == 3:
    #     image_type = "RGB"
    # else:
    #     image_type = "RGBA"

    img = Image.fromarray(
        np.rint(image[..., :3] / max * 255.0).astype(np.int8), mode="RGB"
    )
    return img


viz_rgb = get_rgb_pil_image


def overlay_image(img_1, img_2, alpha=0.5):
    """Overlay two images.

    Args:
        img_1 (PIL.Image): First image.
        img_2 (PIL.Image): Second image.
        alpha (float): Alpha value for blending.
    Returns:
        PIL.Image: Overlayed image.
    """
    return Image.blend(img_1, img_2, alpha=alpha)


def make_gif_from_pil_images(images, filename):
    """Save a list of PIL images as a GIF.

    Args:
        images (list): List of PIL images.
        filename (str): Filename to save GIF to.
    """
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )


def make_video_from_pil_images(images, output_filename, fps=5.0):
    # Generate a random tmp directory name
    tmp_dir = tempfile.mkdtemp()

    # Write files into the tmp directory
    for i, img in enumerate(images):
        img.convert("RGB").save(os.path.join(tmp_dir, "%07d.png" % i))

    subprocess.call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-r",
            str(fps),
            "-i",
            os.path.join(tmp_dir, "%07d.png"),
            output_filename,
        ]
    )


def vstack_images(images, border=10):
    """Stack images vertically.

    Args:
        images (list): List of PIL images.
        border (int): Border between images.
    Returns:
        PIL.Image: Stacked image.
    """
    max_w = 0
    sum_h = (len(images) - 1) * border
    for img in images:
        w, h = img.size
        max_w = max(max_w, w)
        sum_h += h

    full_image = Image.new("RGB", (max_w, sum_h), (255, 255, 255))
    running_h = 0
    for img in images:
        w, h = img.size
        full_image.paste(img, (int(max_w / 2 - w / 2), running_h))
        running_h += h + border
    return full_image


def hstack_images(images, border=10):
    """Stack images horizontally.

    Args:
        images (list): List of PIL images.
        border (int): Border between images.
    Returns:
        PIL.Image: Stacked image.
    """
    max_h = 0
    sum_w = (len(images) - 1) * border
    for img in images:
        w, h = img.size
        max_h = max(max_h, h)
        sum_w += w

    full_image = Image.new("RGB", (sum_w, max_h), (255, 255, 255))
    running_w = 0
    for img in images:
        w, h = img.size
        full_image.paste(img, (running_w, int(max_h / 2 - h / 2)))
        running_w += w + border
    return full_image


def multi_panel(
    images,
    labels=None,
    title=None,
    bottom_text=None,
    title_fontsize=40,
    label_fontsize=30,
    bottom_fontsize=20,
    middle_width=10,
):
    """Combine multiple images into a single image.

    Args:
        images (list): List of PIL images.
        labels (list): List of labels for each image.
        title (str): Title for image.
        bottom_text (str): Text for bottom of image.
        title_fontsize (int): Font size for title.
        label_fontsize (int): Font size for labels.
        bottom_fontsize (int): Font size for bottom text.
        middle_width (int): Width of border between images.
    Returns:
        PIL.Image: Combined image.
    """
    num_images = len(images)
    w = images[0].width
    h = images[0].height

    sum_of_widths = np.sum([img.width for img in images])

    dst = Image.new(
        "RGBA",
        (sum_of_widths + (num_images - 1) * middle_width, h),
        (255, 255, 255, 255),
    )

    drawer = ImageDraw.Draw(dst)
    font_bottom = ImageFont.truetype(
        os.path.join(b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"),
        bottom_fontsize,
    )
    font_label = ImageFont.truetype(
        os.path.join(b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"),
        label_fontsize,
    )
    font_title = ImageFont.truetype(
        os.path.join(b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"),
        title_fontsize,
    )

    bottom_border = 0
    title_border = 0
    label_border = 0
    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        bottom_border = text_h
    if title is not None:
        msg = title
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_title)
        title_border = text_h
    if labels is not None:
        for msg in labels:
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_label)
            label_border = max(text_h, label_border)

    bottom_border += 0
    title_border += 20
    label_border += 20

    dst = Image.new(
        "RGBA",
        (
            sum_of_widths + (num_images - 1) * middle_width,
            h + title_border + label_border + bottom_border,
        ),
        (255, 255, 255, 255),
    )
    drawer = ImageDraw.Draw(dst)

    width_counter = 0
    for j, img in enumerate(images):
        dst.paste(img, (width_counter + j * middle_width, title_border + label_border))
        width_counter += img.width

    if title is not None:
        msg = title
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_title)
        drawer.text(
            (
                (sum_of_widths + (num_images - 1) * middle_width) / 2.0 - text_w / 2,
                title_border / 2 - text_h / 2,
            ),
            msg,
            font=font_title,
            fill="black",
        )

    width_counter = 0
    if labels is not None:
        for i, msg in enumerate(labels):
            w = images[i].width
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_label)
            drawer.text(
                (
                    width_counter + i * middle_width + w / 2 - text_w / 2,
                    title_border + label_border / 2 - text_h / 2,
                ),
                msg,
                font=font_label,
                fill="black",
            )
            width_counter += w

    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        drawer.text(
            (5, title_border + label_border + h + 5),
            msg,
            font=font_bottom,
            fill="black",
        )

    return dst


def make_onehot(n, i, hot=1, cold=0):
    return tuple(cold if j != i else hot for j in range(n))


def multivmap(f, args=None):
    if args is None:
        args = (True,) * len(inspect.signature(f).parameters)
    multivmapped = f
    for i, ismapped in reversed(list(enumerate(args))):
        if ismapped:
            multivmapped = jax.vmap(
                multivmapped, in_axes=make_onehot(len(args), i, hot=0, cold=None)
            )
    return jax.jit(multivmapped)


@jax.jit
def update_choices(trace, addresses, *values):
    return trace.update(
        jax.random.PRNGKey(0),
        genjax.ChoiceMap.from_mapping(zip(addresses.unwrap(), values)),
    )[0]


@jax.jit
def update_choices_get_score(trace, addr_const, *values):
    return update_choices(trace, addr_const, *values).get_score()


enumerate_choices = jax.jit(
    jax.vmap(
        update_choices,
        in_axes=(None, None, 0),
    )
)

enumerate_choices_get_scores = jax.jit(
    jax.vmap(
        update_choices_get_score,
        in_axes=(None, None, 0),
    )
)

grid1 = multivmap(
    update_choices_get_score,
    (
        False,
        False,
        True,
    ),
)
grid2 = multivmap(update_choices_get_score, (False, False, True, True))
grid3 = multivmap(update_choices_get_score, (False, False, True, True, True))
grid4 = multivmap(update_choices_get_score, (False, False, True, True, True, True))


@jax.jit
def grid_trace(trace, addresses_const, values):
    if len(addresses_const.unwrap()) == 1:
        return grid1(trace, addresses_const, *values)
    elif len(addresses_const.unwrap()) == 2:
        return grid2(trace, addresses_const, *values)
    elif len(addresses_const.unwrap()) == 3:
        return grid3(trace, addresses_const, *values)
    elif len(addresses_const.unwrap()) == 4:
        return grid4(trace, addresses_const, *values)
    else:
        raise ValueError("Too many addresses")


def nn_background_segmentation(images):
    import torch
    from carvekit.api.high import HiInterface

    # Check doc strings for more information
    interface = HiInterface(
        object_type="object",  # Can be "object" or "hairs-like".
        batch_size_seg=5,
        batch_size_matting=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False,
    )

    output_images = interface(images)
    masks = jnp.array(
        [jnp.array(output_image)[..., -1] > 0.5 for output_image in output_images]
    )
    return masks


def rr_log_pose(channel, pose, scale=0.1):
    origins = jnp.tile(pose.pos[None, ...], (3, 1))
    colors = jnp.eye(3)
    rr.log(
        channel,
        rr.Arrows3D(
            origins=origins, vectors=pose.as_matrix()[:3, :3].T * scale, colors=colors
        ),
    )


def rr_init(name="demo"):
    rr.init(name)
    rr.connect("127.0.0.1:8812")


def rr_log_rgb(rgb, channel="rgb"):
    rr.log(channel, rr.Image(rgb[..., :3]))


def rr_log_depth(depth, channel="depth"):
    rr.log(channel, rr.DepthImage(depth * 1.0))


def rr_log_mask(mask, channel="mask"):
    rr.log(channel, rr.DepthImage(mask * 1.0))


def rr_log_rgbd(rgbd, channel="rgbd"):
    rr_log_rgb(
        rgbd[..., :3],
        channel + "/rgb",
    )
    rr_log_depth(rgbd[..., 3], channel + "/depth")


def rr_log_cloud(cloud, channel="cloud", colors=None):
    if colors is None:
        rr.log(channel, rr.Points3D(cloud.reshape(-1, 3)))
    else:
        rr.log(channel, rr.Points3D(cloud.reshape(-1, 3), colors=colors.reshape(-1, 3)))


def rr_set_time(t=0):
    rr.set_time_sequence("step", t)


def reload(x):
    import importlib

    importlib.reload(x)


def normalize_log_scores(log_p):
    """
    Normalizes log scores.
    Args:
        log_p: (N,) log scores
    Returns:
        log_p_normalized: (N,) normalized log scores
    """
    return jnp.exp(log_p - jax.scipy.special.logsumexp(log_p))


Array: TypeAlias = jax.Array


def square_center_width_color_to_vertices_faces_colors(i, center, width, color):
    vertices = (
        jnp.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
            ]
        )
        * width
        + center
    )
    faces = (
        jnp.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        + 4 * i
    )
    colors = jnp.ones((4, 3)) * color
    return vertices, faces, colors, jnp.ones(len(faces), dtype=jnp.int32) * i


def all_pairs(X, Y):
    return jnp.swapaxes(
        jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1), 0, 1
    ).reshape(-1, 2)


def distinct_colors(num_colors, pastel_factor=0.5):
    """Get a list of distinct colors.

    Args:
        num_colors (int): Number of colors to generate.
        pastel_factor (float): Pastel factor.
    Returns:
        list: List of colors.
    """
    return [
        np.array(i)
        for i in distinctipy.get_colors(num_colors, pastel_factor=pastel_factor)
    ]


def fit_plane(point_cloud, inlier_threshold, minPoints, maxIteration):
    import pyransac3d

    plane = pyransac3d.Plane()
    plane_eq, _ = plane.fit(
        np.array(point_cloud),
        inlier_threshold,
        minPoints=minPoints,
        maxIteration=maxIteration,
    )
    plane_eq = jnp.array(plane_eq)
    plane_normal = plane_eq[:3]
    point_on_plane = plane_normal * -plane_eq[3]
    plane_x = jnp.cross(plane_normal, np.array([1.0, 0.0, 0.0]))
    plane_y = jnp.cross(plane_normal, plane_x)
    R = jnp.vstack([plane_x, plane_y, plane_normal]).T
    plane_pose = Pose(point_on_plane, Rot.from_matrix(R).as_quat())
    return plane_pose


def fit_table_plane(
    point_cloud, inlier_threshold, segmentation_threshold, minPoints, maxIteration
):
    plane_pose = Pose.fit_plane(point_cloud, inlier_threshold, minPoints, maxIteration)
    points_in_plane_frame = plane_pose.inv().apply(point_cloud)
    inliers = jnp.abs(points_in_plane_frame[:, 2]) < inlier_threshold
    inlier_plane_points = points_in_plane_frame[inliers]

    inlier_table_points_seg = segment_point_cloud(
        inlier_plane_points, segmentation_threshold
    )

    table_points_in_plane_frame = inlier_plane_points[inlier_table_points_seg == 0]

    (cx, cy), (width, height), rotation_deg = cv2.minAreaRect(
        np.array(table_points_in_plane_frame[:, :2])
    )
    pose_shift = Pose(
        jnp.array([cx, cy, 0.0]),
        Rot.from_rotvec(
            jnp.array([0.0, 0.0, 1.0]) * jnp.deg2rad(rotation_deg)
        ).as_quat(),
    )
    table_pose = plane_pose @ pose_shift
    table_dims = jnp.array([width, height, 1e-10])
    return table_pose, table_dims


### Triangle color mesh -> vertex color mesh ###
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


def triangle_color_mesh_to_vertex_color_mesh(vertices, faces, triangle_colors):
    """
    Given a mesh with the provided `vertices, faces, triangle_colors`,
    return an equivalent mesh `(vertices, faces, vertex_colors)`.
    """
    vertices_2, faces_2 = separate_shared_vertices(vertices, faces)
    vertex_colors_2 = jnp.repeat(triangle_colors, 3, axis=0)
    return vertices_2, faces_2, vertex_colors_2


HIINTERFACE = None


def carvekit_get_foreground_mask(image):
    global HIINTERFACE
    if HIINTERFACE is None:
        import torch
        from carvekit.api.high import HiInterface

        HIINTERFACE = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=220,  # 231,
            trimap_dilation=15,
            trimap_erosion_iters=20,
            fp16=False,
        )
    imgs = HIINTERFACE([b3d.get_rgb_pil_image(image)])
    mask = jnp.array(imgs[0])[..., -1] > 0.5
    return mask


def discretize(data, resolution):
    """
    Discretizes a point cloud.
    """
    return jnp.round(data / resolution) * resolution


def voxelize(data, resolution):
    """
        Voxelize a point cloud.
    Args:
        data: (N,3) point cloud
        resolution: (float) resolution of the voxel grid
    Returns:
        data: (M,3) voxelized point cloud
    """
    data = discretize(data, resolution)
    data, indices, occurences = jnp.unique(
        data, axis=0, return_index=True, return_counts=True
    )
    return data, indices, occurences


@jax.jit
def voxel_occupied_occluded_free(
    camera_pose, rgb_image, depth_image, grid, fx, fy, cx, cy, far, tolerance
):
    grid_in_cam_frame = camera_pose.inv().apply(grid)
    height, width = depth_image.shape[:2]
    pixels = b3d.xyz_to_pixel_coordinates(grid_in_cam_frame, fx, fy, cx, cy).astype(
        jnp.int32
    )
    valid_pixels = (
        (0 <= pixels[:, 0])
        * (0 <= pixels[:, 1])
        * (pixels[:, 0] < height)
        * (pixels[:, 1] < width)
    )
    real_depth_vals = depth_image[pixels[:, 0], pixels[:, 1]] * valid_pixels + (
        1 - valid_pixels
    ) * (far + 1.0)

    projected_depth_vals = grid_in_cam_frame[:, 2]
    occupied = jnp.abs(real_depth_vals - projected_depth_vals) < tolerance
    real_rgb_values = rgb_image[pixels[:, 0], pixels[:, 1]] * occupied[..., None]
    occluded = real_depth_vals < projected_depth_vals
    occluded = occluded * (1.0 - occupied)
    _free = (1.0 - occluded) * (1.0 - occupied)
    return 1.0 * occupied - 1.0 * _free, real_rgb_values


voxel_occupied_occluded_free_parallel_camera = jax.jit(
    jax.vmap(
        voxel_occupied_occluded_free,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
)
voxel_occupied_occluded_free_parallel_camera_depth = jax.jit(
    jax.vmap(
        voxel_occupied_occluded_free,
        in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    )
)


def make_grid_points(min_vec, max_vec, num_vec):
    """
    Generate uniformly spaced translation proposals in a 3D box
    Args:
        min_x, min_y, min_z: minimum x, y, z values
    """
    deltas = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.linspace(min_vec[i], max_vec[i], num_vec[i])
                for i in range(len(min_vec))
            ],
        ),
        axis=-1,
    )
    deltas = deltas.reshape((-1, len(min_vec)), order="F")
    return deltas
