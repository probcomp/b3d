import numpy as np
import cv2
import imageio
from b3d.io.feature_track_data import FeatureTrackData
from b3d.io.segmented_video_input import SegmentedVideoInput


""" Utils """


def add_label(img, label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = img.shape[0] / 400
    thickness = img.shape[0] // 200
    label_height = img.shape[0] // 10
    label_position = img.shape[0] // 14
    color = (255, 255, 255)
    img = cv2.copyMakeBorder(
        img, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    cv2.putText(
        img,
        label,
        (10, label_position),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return img


def resize_img(img, downscale):
    return cv2.resize(img, (0, 0), fx=downscale, fy=downscale)


def create_color_map(segmentation):
    unique_ids = np.unique(segmentation)
    # color_map = np.zeros((len(unique_ids), 3), dtype=np.uint8)
    color_map = np.zeros((np.max(unique_ids) + 1, 3), dtype=np.uint8)

    np.random.seed(42)
    for uid in unique_ids:
        color_map[uid] = np.random.randint(0, 255, 3)

    return color_map


""" Image formatting """


def create_rgb_image(rgb, label=None):
    if rgb.dtype == np.uint8:
        rgb_img = np.array(rgb, dtype=np.uint8)
    elif rgb.dtype == np.float32:
        rgb_img = np.array(rgb * 255, dtype=np.uint8)
    else:
        raise ValueError("Unsupported RGB data type")

    if label:
        rgb_img = add_label(rgb_img, label)
    return rgb_img.astype(np.uint8)


def create_depth_image(depth_float, label=None):
    depth_img = np.array(depth_float, dtype=float)
    depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_img = cv2.applyColorMap(depth_img.astype(np.uint8), cv2.COLORMAP_JET)
    if label:
        depth_img = add_label(depth_img, label)
    return depth_img.astype(np.uint8)


def create_keypoints_image(visible_keypoints, W, H, label=None):
    keypoints_img = np.zeros((H, W, 3), dtype=np.uint8)
    for kp in visible_keypoints:
        cv2.circle(keypoints_img, (int(kp[0]), int(kp[1])), 4, (0, 255, 0), -1)
    if label:
        keypoints_img = add_label(keypoints_img, label)
    return keypoints_img.astype(np.uint8)


def create_segmentation_image(segmentation_int, color_map, label=None):
    segmentation_img = color_map[segmentation_int].astype(np.uint8)
    if label:
        segmentation_img = add_label(segmentation_img, label)
    return segmentation_img.astype(np.uint8)


""" Single data type videos """


def create_video(
    img_array,
    create_image_function,
    output_path="output.gif",
    label=None,
    res=None,
    fps=10,
    slow=1,
    source_fps=30,
):
    Nframe = img_array.shape[0]
    frames = []

    width = img_array.shape[2]
    resize_factor = 1
    if res:
        resize_factor = float(res / width)
    frame_step = int(source_fps / fps)

    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        frame = create_image_function(img_array[t], label=label)
        frame = resize_img(frame, resize_factor)
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=slow * fps)
    print(f"Saved {output_path}")


def create_keypoints_video(
    keypoints_positions,
    visibility_mask,
    width,
    height,
    output_path="output.gif",
    label=None,
    res=None,
    fps=10,
    slow=1,
    source_fps=30,
):
    Nframe = keypoints_positions.shape[0]
    frames = []

    resize_factor = 1
    if res:
        resize_factor = float(res / width)
    frame_step = int(source_fps / fps)

    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        visible_keypoints = keypoints_positions[t][visibility_mask[t]]
        frame = create_keypoints_image(visible_keypoints, width, height, label=label)
        frame = resize_img(frame, resize_factor)
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=slow * fps)
    print(f"Saved {output_path}")


def create_segmentation_video(
    img_array,
    output_path="output.gif",
    res=None,
    fps=10,
    slow=1,
    source_fps=30,
    label=None,
):
    Nframe = img_array.shape[0]
    frames = []

    width = img_array.shape[2]
    resize_factor = 1
    if res:
        resize_factor = float(res / width)
    frame_step = int(source_fps / fps)

    color_map = create_color_map(img_array)

    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        frame = create_segmentation_image(
            img_array[t], color_map=color_map, label=label
        )
        frame = resize_img(frame, resize_factor)
        frames.append(frame)

    imageio.mimsave(output_path, frames, fps=slow * fps)
    print(f"Saved {output_path}")


""" Combined images"""


def create_feature_track_frame(rgb, depth, visible_keypoints, width=None, height=None):
    if width is None:
        width = int(rgb.shape[1])
    if height is None:
        height = int(rgb.shape[0])
    rgb_img = create_rgb_image(rgb, "RGB")
    depth_img = create_depth_image(depth, "Depth")
    keypoints_img = create_keypoints_image(
        visible_keypoints, width, height, "Visible Keypoints"
    )
    img = np.hstack((rgb_img, depth_img, keypoints_img))
    return img.astype(np.uint8)


def create_segmented_video_input_frame(rgb, depth, segmentation, color_map):
    rgb_img = create_rgb_image(rgb, "RGB")
    depth_img = create_depth_image(depth, "Depth")
    segmentation_img = create_segmentation_image(
        segmentation, color_map, "Segmentation"
    )
    img = np.hstack((rgb_img, depth_img, segmentation_img))
    return img.astype(np.uint8)


""" Combined videos """


def create_feature_track_video(
    data: FeatureTrackData,
    output_path="output.gif",
    res=None,
    fps=10,
    slow=1,
    source_fps=30,
):
    Nframe = data.rgbd_images.shape[0]
    width = data.rgbd_images.shape[2]
    height = data.rgbd_images.shape[1]

    resize_factor = 1
    if res:
        resize_factor = float(res / width)

    frames = []
    frame_step = int(source_fps / fps)
    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        rgb = data.rgbd_images[t, :, :, :3]
        depth = data.rgbd_images[t, :, :, 3]
        kp = data.observed_keypoints_positions[t][data.keypoint_visibility[t]]
        frame = create_feature_track_frame(rgb, depth, kp, width, height)
        frame = resize_img(frame, resize_factor)
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=slow * fps)
    print(f"Saved {output_path}")


def create_segmented_video_input_video(
    data: SegmentedVideoInput,
    output_path="output.gif",
    res=None,
    fps=10,
    slow=1,
    source_fps=30,
):
    Nframe = data.rgb.shape[0]

    width = data.rgb.shape[2]
    resize_factor = 1
    if res:
        resize_factor = float(res / width)

    color_map = create_color_map(data.segmentation)

    frames = []
    frame_step = int(source_fps / fps)
    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        rgb = data.rgb[t, :, :, :3]

        # Compute depth image from XYZ positions
        xyz = data.xyz[t]
        depth = xyz[..., 2]  # Z is the depth
        segmentation = data.segmentation[t]
        frame = create_segmented_video_input_frame(rgb, depth, segmentation, color_map)
        frame = resize_img(frame, resize_factor)
        frames.append(frame)
    imageio.mimsave(output_path, frames, fps=slow * fps)
    print(f"Saved {output_path}")
