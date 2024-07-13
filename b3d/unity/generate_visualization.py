import numpy as np
import cv2
import imageio
from b3d.io.feature_track_data import FeatureTrackData

def create_keypoints_gif(data: FeatureTrackData, output_path='output.gif', res: float=200, fps: float=10, slow_gif: float=1):
    Nframe = data.rgbd_images.shape[0]
    frames = []

    width = data.camera_intrinsics[0]
    resize_factor = float(res / width)
    frame_step = int(data.fps / fps)

    # Define label properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = 0.5
    base_thickness = 1
    base_circle_radius = 1
    color = (255, 255, 255)

    # Rescale properties 
    font_scale = base_font_scale / resize_factor
    thickness = int(base_thickness / resize_factor)
    circle_radius = int(base_circle_radius / resize_factor)
    label_height = int(20 / resize_factor)
    label_position = int(label_height // 1.4)

    for t in range(0, Nframe, frame_step):  # Skip every "frame_step"
        rgb_image = np.array(data.rgbd_images[t, :, :, :3] * 255, dtype=np.uint8)
        depth_image = np.array(data.rgbd_images[t, :, :, 3], dtype=float)

        # Normalize depth image for visualization
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)

        keypoints_image = np.zeros_like(rgb_image)
        visible_keypoints = data.observed_keypoints_positions[t][data.keypoint_visibility[t]]
        for kp in visible_keypoints:
            cv2.circle(keypoints_image, (int(kp[0]), int(kp[1])), circle_radius, (0, 255, 0), -1)

        # Add labels
        labeled_rgb = cv2.copyMakeBorder(rgb_image, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        labeled_depth = cv2.copyMakeBorder(depth_image, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        labeled_keypoints = cv2.copyMakeBorder(keypoints_image, label_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        cv2.putText(labeled_rgb, 'RGB', (10, label_position), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(labeled_depth, 'Depth', (10, label_position), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(labeled_keypoints, 'Visible Keypoints', (10, label_position), font, font_scale, color, thickness, cv2.LINE_AA)

        combined_image = np.hstack((labeled_rgb, labeled_depth, labeled_keypoints))

        # Resize the combined image
        combined_image = cv2.resize(combined_image, (0, 0), fx=resize_factor, fy=resize_factor)

        frames.append(combined_image)

    # Adjust fps to account for skipping frames
    imageio.mimsave(output_path, frames, fps=fps * slow_gif)