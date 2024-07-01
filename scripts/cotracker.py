import torch
import b3d
import os
import numpy as np
import time
import argparse
from pathlib import Path
from b3d.io.utils import add_argparse, path_stem
from b3d.io import FeatureTrackData

_cotracker_info = """
Source path:
    {source_path}
Target path:
    {target_path}
Grid size:
    {grid_size} x {grid_size}
"""

def _cotracker(source_path, target_dir=None, grid_size=50):
    """
    Run CoTracker on a video input data file and save the results.
    """
    source_path = Path(source_path)
    if target_dir is None: target_dir = source_path.parent
    else: target_dir = Path(target_dir)
    target_path = target_dir / f"{path_stem(source_path)}.FeatureTrackData.npz"
    grid_size = int(grid_size)

    print(_cotracker_info.format(
        source_path=source_path, target_path=target_path, grid_size=grid_size))

    video_input = b3d.io.VideoInput.load(source_path)
    frames = np.array(video_input.rgb)

    print(f"Frames shape: {frames.shape}")

    device = 'cuda'
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)

    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
    t0 = time.time()
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
    t1 = time.time()

    print(f"Cotracker took {t1-t0:.2f} seconds.")

    pred_tracks_ = pred_tracks.cpu().numpy()
    pred_visibility_ = pred_visibility.cpu().numpy()

    ftd = FeatureTrackData(
        observed_keypoints_positions =  pred_tracks_[0],
        keypoint_visibility = pred_visibility_[0],
        rgbd_images =  video_input.rgbd,
        camera_intrinsics =  video_input.camera_intrinsics_rgb,
        fps = video_input.fps)

    ftd.save(target_path)


@add_argparse
def cotracker(source_path, target_dir=None, grid_size=50):
    """
    Run CoTracker on a `VideoInput` file and save the results.
    """
    return _cotracker(source_path, target_dir, grid_size)


if __name__ == "__main__":
    cotracker()
