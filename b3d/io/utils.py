import os
from .video_input import VideoInput
from b3d.utils import get_shared
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import cv2
import numpy as np
from sklearn.utils import Bunch
from pathlib import Path
import argparse
import inspect
import sys


def add_argparse(f):
    """
    Decorator that automatically adds an argument parser 
    from `f`'s signature, so it can be used from the command line.
    """
    parser = argparse.ArgumentParser(description=f.__doc__)
    sig = inspect.signature(f)
    
    for k,v in sig.parameters.items():
        # FYI: you get the type annotation
        # via `sig.parameters[k].annotation`
        
        if v.default is inspect.Parameter.empty:
            parser.add_argument(k)  
        else:
            parser.add_argument(f"-{k}", f"--{k}", 
                                default=v.default) 
            
    def g():
        args = parser.parse_args()
        f(**vars(args))
        
    return g


def path_stem(path):
    """Removes ALL suffixes from a path."""
    name = Path(path).name
    for _ in path.suffixes:
        name = name.rsplit('.')[0]

    return name

_video_summary = """
File: \033[96m{info.fname.name}\033[0m
- T: \033[1m{info.timesteps}\033[0m
- h: \033[1m{info.height}\033[0m
- w: \033[1m{info.width}\033[0m
- fps: \033[1m{info.fps}\033[0m
"""
class VideoInfo(Bunch):
    def __str__(self): return _video_summary.format(info=self)
    def __repr__(self): return self.__str__()

    @property
    def T(self): return self.timesteps
    @property
    def w(self): return self.width  
    @property
    def h(self): return self.height



def load_video_info(file_path):
    """
    Returns a Bunch object containing information about the video file, i.e. 
    `timesteps`, `width`, `height`, and `fps`.
    """
    # Open the video file
    cap = cv2.VideoCapture(str(file_path))

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None


    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Release the video capture object
    cap.release()

    return VideoInfo(timesteps=T, width=w, height=h, fps=fps, fname=Path(file_path))


def load_video_to_numpy(file_path, step=1, times=None, downsize=1, reverse_color_channel=False):
    """
    Read video file and convert to numpy array.

    Args:
        `file_path` (str): Path to video file.
        `step` (int): Read every `step` frame from the video.
        `downsize` (int): Downsize the video by a factor of `downsize`.
        `reverse_color_channel` (bool): Reverse the color channel of the video.

    Example:
    ```Python
    video_array = load_video_to_numpy('_test_vid.mp4')
    ```
    """
    # Open the video file
    cap = cv2.VideoCapture(str(file_path))

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None


    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if times is None: times = np.arange(T, step=step) 
    t = -1
    frames = []
    while True:
        # Potentially skip frames
        # Read frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret: break

        t += 1
        if t not in times:
            continue
        
        # Resize if necessary, and 
        # add it to the list
        if downsize > 1:
            frame = cv2.resize(frame, dsize=(w//downsize, h//downsize), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)


    # Release the video capture object
    cap.release()

    # Convert list of frames to numpy array
    video_array = np.array(frames)
    if reverse_color_channel:
        video_array = video_array[..., ::-1]

    return video_array


def video_input_from_mp4(
        video_fname, 
        intrinsics_fname, 
        step=1, 
        times=None, 
        downsize=1, 
        reverse_color_channel=False):
    if times is None: times = np.arange(T, step=step) 

    info = load_video_info(video_fname)
    intr = np.load(intrinsics_fname, allow_pickle=True)
    vid  = load_video_to_numpy(video_fname, 
            times=times, 
            downsize=downsize, 
            reverse_color_channel=reverse_color_channel)

    fps = info.fps/(times[1] - times[0])
    
    return VideoInput(rgb=vid, camera_intrinsics_rgb=intr, fps=fps)


def plot_video_summary(
        video_fname, 
        start=0, 
        end=None, 
        reverse_color_channel=False, 
        downsize=10, 
        num_summary_frames=10):
    """
    Plots a summary of the video frames.
    """
    # TODO: Add support for other video formats AND video input instances
    if video_fname.suffix not in [".mp4"]:
        raise ValueError(f"Only .mp4 files are supported. Got: {video_fname.suffix}")


    info = load_video_info(video_fname)

    T0 = start or 0
    T1 = end or info.timesteps
    times = np.linspace(T0,T1-1, num_summary_frames).astype(int)  

    vid = load_video_to_numpy(video_fname, times=times, downsize=downsize, reverse_color_channel=reverse_color_channel)

    w = vid.shape[2]
    h = vid.shape[1]

    # Create a plot with the summary
    # TODO: Should we hand in an axis?
    fig, ax = plt.subplots(1,1, figsize=(15,4))
    ax.set_title(f"\"{video_fname.name}\"\n(start = {T0}, end = {T1}, fps = {info.fps})")
    ax.imshow(np.concatenate(vid, axis=1))
    for i,t in enumerate(times):
        ax.text(i*w + 7, h - 7, f"{t}", size=11, rotation=0.,
                ha="left", va="bottom",
                bbox=dict(boxstyle="square",
                        ec=(1., 1., 1., 0.),
                        fc=(1., 1., 1., 1.),
                        )
                )
    ax.axis("off");


