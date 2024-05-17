import jax.numpy as jnp
from functools import partial
import numpy as np
from collections import namedtuple
import genjax
from PIL import Image
import subprocess
import jax
import sklearn.cluster

import inspect
from inspect import signature
import genjax
import b3d
from pathlib import Path
import os
import trimesh
from b3d import Pose
import rerun as rr

from dataclasses import dataclass


def get_root_path() -> Path:
    return Path(Path(b3d.__file__).parents[1])


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


def get_shared() -> Path:
    """The absolute path of the assets directory on current machine"""
    data_dir_path = get_assets() / "shared_data_bucket"

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        print(f"Initialized empty directory for shared bucket data at {data_dir_path}.")

    return data_dir_path


def get_gcloud_bucket_ref() -> str:
    return "gs://hgps_data_bucket"


def xyz_from_depth(z: "Depth Image", fx, fy, cx, cy):
    v, u = jnp.mgrid[: z.shape[0], : z.shape[1]]
    x = (u - cx) / fx
    y = (v - cy) / fy
    xyz = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * z[..., None]
    return xyz

def xyz_to_pixel_coordinates(xyz, fx, fy, cx, cy):
    x = fx * xyz[..., 0] / xyz[..., 2] + cx
    y = fy * xyz[..., 1] / xyz[..., 2] + cy
    return jnp.stack([y, x], axis=-1)


@partial(jnp.vectorize, signature="(k)->(k)")
def rgb_to_lab(rgb):
    # Convert sRGB to linear RGB
    rgb = jnp.clip(rgb, 0, 1)
    mask = rgb > 0.04045
    rgb = jnp.where(mask, jnp.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)

    # RGB to XYZ
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    rgb_to_xyz = jnp.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = jnp.dot(rgb, rgb_to_xyz.T)

    # XYZ to LAB
    # https://en.wikipedia.org/wiki/CIELAB_color_space#From_CIEXYZ_to_CIELAB
    xyz_ref = jnp.array([0.95047, 1.0, 1.08883])  # D65 white point
    xyz_normalized = xyz / xyz_ref
    mask = xyz_normalized > 0.008856
    xyz_f = jnp.where(
        mask, jnp.power(xyz_normalized, 1 / 3), 7.787 * xyz_normalized + 16 / 116
    )

    L = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])

    lab = jnp.stack([L, a, b], axis=-1)
    return lab


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


def make_mesh_from_point_cloud_and_resolution(grid_centers, grid_colors, resolutions):
    box_mesh = trimesh.creation.box(jnp.ones(3))
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


def get_rgb_pil_image(image, max=1.0):
    """Convert an RGB image to a PIL image.

    Args:
        image (np.ndarray): RGB image. Shape (H, W, 3).
        max (float): Maximum value for colormap.
    Returns:
        PIL.Image: RGB image visualized as a PIL image.
    """
    image = np.clip(image, 0.0, max)
    if image.shape[-1] == 3:
        image_type = "RGB"
    else:
        image_type = "RGBA"

    img = Image.fromarray(
        np.rint(image / max * 255.0).astype(np.int8),
        mode=image_type,
    ).convert("RGB")
    return img


import tempfile
import subprocess
import os
def make_video_from_pil_images(images, output_filename, fps=5.0):
    # Generate a random tmp directory name
    tmp_dir = tempfile.mkdtemp()

    # Write files into the tmp directory
    for i, img in enumerate(images):
        img.convert("RGB").save(os.path.join(tmp_dir, "%07d.png" % i))

    subprocess.call(["ffmpeg", "-hide_banner", "-loglevel",  "error", "-y", "-r", str(fps), "-i", os.path.join(tmp_dir, "%07d.png"), output_filename])


from PIL import Image, ImageDraw, ImageFont

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
        os.path.join(
            b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"
        ),
        bottom_fontsize,
    )
    font_label = ImageFont.truetype(
        os.path.join(
            b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"
        ),
        label_fontsize,
    )
    font_title = ImageFont.truetype(
        os.path.join(
            b3d.get_assets(), "fonts", "IBMPlexSerif-Regular.ttf"
        ),
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
    return multivmapped


def update_choices(trace, key, addr_const, *values):
    addresses = addr_const.const
    return trace.update(
        key,
        genjax.choice_map({addr: c for (addr, c) in zip(addresses, values)}),
        genjax.Diff.tree_diff_unknown_change(trace.get_args()),
    )[0]


update_choices_jit = jax.jit(update_choices, static_argnums=(2,))


def update_choices_get_score(trace, key, addr_const, *values):
    return update_choices(trace, key, addr_const, *values).get_score()


update_choices_get_score_jit = jax.jit(update_choices_get_score, static_argnums=(2,))

enumerate_choices = jax.vmap(
    update_choices,
    in_axes=(None, None, None, 0),
)
enumerate_choices_jit = jax.jit(enumerate_choices, static_argnums=(2,))

enumerate_choices_get_scores = jax.vmap(
    update_choices_get_score,
    in_axes=(None, None, None, 0),
)
enumerate_choices_get_scores_jit = jax.jit(
    enumerate_choices_get_scores, static_argnums=(2,)
)


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


def rr_log_pose(channel, pose):
    origins = jnp.tile(pose.pos[None, ...], (3, 1))
    vectors = jnp.eye(3)
    colors = jnp.eye(3)
    rr.log(
        channel,
        rr.Arrows3D(origins=origins, vectors=pose.as_matrix()[:3, :3].T, colors=colors),
    )


def normalize_log_scores(log_p):
    """
    Normalizes log scores.
    Args:
        log_p: (N,) log scores
    Returns:
        log_p_normalized: (N,) normalized log scores
    """
    return jnp.exp(log_p - jax.scipy.special.logsumexp(log_p))


from typing import Any, NamedTuple, TypeAlias
import jax

Array: TypeAlias = jax.Array


@dataclass
class VideoInput:
    """
    Video data input. Note: Spatial units are measured in meters.

    World Coordinates. The floor is x,y and up is z.
    Camera Pose. The camera pose should be interpretted as the z-axis pointing out of the camera,
        x-axis pointing to the right, and y-axis pointing down. This is the OpenCV convention.
    Quaternions. We follow scipy.spatial.transform.Rotation.from_quat which uses scalar-last (x, y, z, w)
    Camera Intrinsics. We store it as an array of shape (8,) containing width, height, fx, fy, cx, cy, near, far.
        The camera matrix is given by: $$ K = \begin{bmatrix} f_x & 0 & c_x \ 0 & f_y & c_y \ 0 & 0 & 1 \end{bmatrix} $$
    Spatial units. Spatial units are measured in meters (if not indicated otherwise).

    **Attributes:**
    - rgb
        video_input['rgb'][t,i,j] contains RGB values in the interval [0,255] of pixel i,j at time t.
        Shape: (T,H,W,3) -- Note this might be different from the width and height of xyz
        Type: uint8, in [0,255]
    - xyz
        video_input['xyz'][t,i,j] is the 3d point associated with pixel i,j at time t in camera coordinates
        Shape: (T, H', W', 3) -- Note this might be different from the width and height of rgb
        Type: Float
    - camera_positions
        video_input['camera_positions'][t] is the position of the camera at time t
        Shape: (T, 3)
        Type: Float
    - camera_quaternions
        video_input['camera_quaternions'][t] is the quaternion (in xyzw format) representing the orientation of the camera at time t
        Shape: (T, 4)
        Type: Float
    - camera_intrinsics_rgb
        video_input['camera_intrinsics_rgb'][:] contains width, height, fx, fy, cx, cy, near, far. Width and height determine the shape of rgb above
        Shape: (8,)
        Type: Float
    - camera_intrinsics_depth
        video_input['camera_intrinsics_depth'][:] contains width, height, fx, fy, cx, cy, near, far. Width and height determine the shape of xyz above
        Shape: (8,)
        Type: Float

    **Note:**
    For compactness, rgb values are saved as uint8 values, however
    the output of the renderer is a float between 0 and 1. VideoInput
    stores uint8 colors, so please use the rgb_float property for
    compatibility.

    **Note:**
    The width and height of the `rgb` and `xyz` arrays may differ.
    Their shapes match the entries in `camera_intrinsics_rgb` and
    `camera_intrinsics_depth`, respectively. The latter was used
    to project the `depth` arrays to `xyz`.
    """

    rgb: Array  # [num_frames, height_rgb, width_rgb, 3]
    xyz: Array  # [num_frames, height_depth, width_depth, 3]
    camera_positions: Array  # [num_frames, 3]
    camera_quaternions: Array  # [num_frames, 4]
    camera_intrinsics_rgb: (
        Array  # [8,] (width_rgb, height_rgb, fx, fy, cx, cy, near, far)
    )
    camera_intrinsics_depth: (
        Array  # [8,] (width_depth, height_depth, fx, fy, cx, cy, near, far)
    )

    def __post_init__(self):
        super().__init__()
        assert self.rgb.shape[0] == self.xyz.shape[0]
        assert self.rgb.shape[1] == self.camera_intrinsics_rgb[1]
        assert self.rgb.shape[2] == self.camera_intrinsics_rgb[0]
        assert self.rgb.dtype == jnp.uint8
        assert len(self.xyz.shape) == 4
        assert len(self.rgb.shape) == 4
        assert self.rgb.shape[-1] == 3
        assert self.xyz.shape[-1] == 3

    def to_dict(self):
        return {
            "rgb": self.rgb,
            "xyz": self.xyz,
            "camera_positions": self.camera_positions,
            "camera_quaternions": self.camera_quaternions,
            "camera_intrinsics_rgb": self.camera_intrinsics_rgb,
            "camera_intrinsics_depth": self.camera_intrinsics_depth,
        }

    def save(self, filepath: str):
        """Saves VideoInput to file"""
        jnp.savez(
            filepath,
            rgb=self.rgb,
            xyz=self.xyz,
            camera_positions=self.camera_positions,
            camera_quaternions=self.camera_quaternions,
            camera_intrinsics_rgb=self.camera_intrinsics_rgb,
            camera_intrinsics_depth=self.camera_intrinsics_depth,
        )

    def save_in_timeframe(self, filepath: str, start_t: int, end_t: int):
        """Saves new VideoInput containing data
        between a timeframe into file"""
        jnp.savez(
            filepath,
            rgb=self.rgb[start_t:end_t],
            xyz=self.xyz[start_t:end_t],
            camera_positions=self.camera_positions[start_t:end_t],
            camera_quaternions=self.camera_quaternions[start_t:end_t],
            camera_intrinsics_rgb=self.camera_intrinsics_rgb,
            camera_intrinsics_depth=self.camera_intrinsics_depth,
        )

    @classmethod
    def load(cls, filepath: str):
        """Loads VideoInput from file"""
        with open(filepath, "rb") as f:
            data = jnp.load(f, allow_pickle=True)
            return cls(
                rgb=jnp.array(data["rgb"]),
                xyz=jnp.array(data["xyz"]),
                camera_positions=jnp.array(data["camera_positions"]),
                camera_quaternions=jnp.array(data["camera_quaternions"]),
                camera_intrinsics_rgb=jnp.array(data["camera_intrinsics_rgb"]),
                camera_intrinsics_depth=jnp.array(data["camera_intrinsics_depth"]),
            )

    @property
    def rgb_float(self):
        if self.rgb.dtype == jnp.uint8:
            return self.rgb / 255.0
        else:
            return self.rgb


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
        jnp.stack(jnp.meshgrid(jnp.arange(X), jnp.arange(Y)), axis=-1),
        0, 1
    ).reshape(-1, 2)
