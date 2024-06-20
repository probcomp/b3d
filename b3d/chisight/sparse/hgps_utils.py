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


def load_video_to_numpy(file_path):
    """
    Read video file and convert to numpy array.

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

    frames = []
    while True:
        # Read frame-by-frame
        ret, frame = cap.read()

        # If the frame was read successfully, add it to the list
        if ret:
            frames.append(frame)
        else:
            break

    # Release the video capture object
    cap.release()

    # Convert list of frames to numpy array
    video_array = np.array(frames)

    return video_array
