from warnings import warn
from functools import wraps
import jax
import cv2
import numpy as np


def depreciated(msg):
    """
    Decorator to mark functions as depreciated.

    Example Usage:
    ```
    @depreciated("Use `something_new` instead.")
    def something_old(x):
        return x
    ```

    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            warn(
                f"Call to DEPRECIATED func `{f.__name__}`...{msg}",
                DeprecationWarning,
                stacklevel=2,
            )
            return f(*args, **kwargs)

        return wrapper

    return decorator


def load_video_to_numpy(file_path, step=1, downsize=1, reverse_color_channel=False):
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
    fps = cap.get(cv2.CAP_PROP_FPS)

    t = -1
    frames = []
    while True:
        # Potentially skip frames
        # Read frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret: break

        t += 1
        if t % step != 0:
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
