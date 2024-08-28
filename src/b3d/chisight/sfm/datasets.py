import subprocess
import os
from pathlib import Path
import imageio.v3 as iio
import re
from b3d.camera import Intrinsics
from b3d.pose import Pose
import numpy as np
import jax
import jax.numpy as jnp


_DOWNLOAD_BASH_SCRIPT = """#!/bin/bash

# Check if both sequence and target_folder arguments are provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 <sequence_url> <target_folder>"
  exit 1
fi

# Assign arguments to variables
sequence_url=$1
target_folder=$2
sequence=$(basename $sequence_url)

echo "Downloading $sequence to $target_folder..."

# Ensure the target folder exists
mkdir -p "$target_folder"

# Download the file using wget
wget "$sequence_url" -P "$target_folder" 

# Extract the tar.gz file
tar -xzf "$target_folder/$sequence" -C "$target_folder"

# Remove the tar.gz file
rm "$target_folder/$sequence"
"""


class RgbdSlamSequenceData:
    """"
    Helper class to handle RGB-D Sequences from the TUM RGBD SLAM benchmark dataset.
    The dataset can be downloaded from the following link:
    > https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
    
    Example Usage:
    ```
    # Grab a sequence URL from set
    # a target folder to store the data 
    # > https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
    sequence_url  = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
    target_folder = "~/workspace/rgbd_slam_dataset_freiburg"

    # Download and extract the sequence data into 
    # a new folder under the target folder
    sequence_folder = RgbdSlamSequenceData._download_from_url(sequence_url, target_folder)
    data = RgbdSlamSequenceData(sequence_folder)

    # Get the i'th RGB image
    # Note that rgb, depth, and pose sequences are not synchronized, so the i'th RGB image
    # and the i'th depth image and pose are not guaranteed to be from the same time.
    i = 100
    rgb = data.get_rgb(i)

    # This returns i'th RGB image and the CLOSEST (in time) available depth image and pose
    rgb, depth, pose = data.get_synced(i)

    # Plot the RGB and depth images side by side
    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    axs[0].imshow(rgb)
    axs[1].imshow(np.where(depth>0, depth, np.nan))
    axs[2].imshow(rgb, alpha=1.)
    axs[2].imshow(np.where(depth>0, depth, np.nan), alpha=0.75)
    ```
    """
    def __init__(self, path):
        """
        Args:
            path (str): Path to one ofthe the TUM RGB-D datasets, e.g., 
                .../data/rgbd_dataset_freiburg2_desk
        """

        self.path = path

        self.gt_data = np.loadtxt(path/"groundtruth.txt", comments='#', dtype = [
            ('timestamp', 'f8'), 
            ('tx', 'f8'), ('ty', 'f8'), ('tz', 'f8'), 
            ('qx', 'f8'), ('qy', 'f8'), ('qz', 'f8'), ('qw', 'f8')])

        self.rgb_data = np.loadtxt(path/"rgb.txt", comments='#', dtype=[
            ("timestamp", 'f8'), ("filename", 'U50')])

        self.depth_data = np.loadtxt(path/"depth.txt", comments='#', dtype=[
            ("timestamp", 'f8'), ("filename", 'U50')])
    
    @staticmethod
    def _download_from_url(sequence_url, target_folder):

        # Target folder for the sequence data
        sequence_folder = Path(target_folder)/Path(sequence_url).stem

        # Check if the target folder exists
        if os.path.exists(sequence_folder):
            print(f"Sequence \n\t\"{Path(sequence_url).stem}\"\nalready exists here \n\t{sequence_folder}")
            return sequence_folder

        try:
            # Execute the Bash script using subprocess and pass in the arguments
            print("Downloading and extracting...this might take a minute....")
            result = subprocess.run(
                ['bash', '-c', _DOWNLOAD_BASH_SCRIPT, '_', sequence_url, target_folder],
                check=True,
                text=True,
                capture_output=True
            )

            # Print the output of the script
            print("Script Output:...\n", result.stdout)
            print("Script executed successfully.")
            return sequence_folder

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the script: {e}")
            print(f"Error output: {e.stderr}")

    def len(self):
        """Returns the number of (RGB) frames in the dataset."""
        return len(self.rgb_data)
    
    def shape(self):
        """Returns the shape of the RGB images."""
        return self.get_rgb(0).shape

    def get_rgb(self, i):
        """Returns the RGB image at index i."""
        return iio.imread(self.path/self.rgb_data[i][1])

    def get_depth(self, i):
        """Returns the depth image at index i."""
        return iio.imread(self.path/self.depth_data[i][1])/5_000
    
    def get_pose(self, i):
        """Returns the pose at index i."""
        _, tx, ty, tz, qx, qy, qz, qw = self.gt_data[i]
        return Pose(jnp.array([tx,ty,tz]), jnp.array([qx, qy, qz, qw]))
    
    def get_synced(self, i):
        """Returns the timestamp, RGB, depth image, and pose at index i."""
        t = self.rgb_data[i]["timestamp"]
        i_pose = np.argmin(np.abs(self.gt_data["timestamp"] - t))
        i_depth = np.argmin(np.abs(self.depth_data["timestamp"] - t))
        return self.get_rgb(i), self.get_depth(i_depth), self.get_pose(i_pose)
    
    def get_timestamp(self, i):
        return self.rgb_data[i]["timestamp"]
    
    def __getitem__(self, i):
        """Returns the RGB, depth image, and pose at index i."""
        return self.get_synced(i)

    def get_intrinsics(self, index=0):
        """Returns the camera intrinsics."""
        # See 
        # > https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
        intr0 = Intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5, 1e-2, 1e-4)
        intr1 = Intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3, 1e-2, 1e-4)
        intr2 = Intrinsics(640, 480, 520.9, 521.0, 325.1, 249.7, 1e-2, 1e-4)
        intr3 = Intrinsics(640, 480, 535.4, 539.2, 320.1, 247.6, 1e-2, 1e-4)

        return [intr0, intr1, intr2, intr3][index]


def val_from_im(uv, im):
    return im[int(uv[1]), int(uv[0])]


vals_from_im = jax.vmap(val_from_im, in_axes=(0, None))


def _extract_number_after_freiburg(input_string):
    match = re.search(r'freiburg(\d+)', input_string)
    if match:
        return int(match.group(1))
    else:
        return None


def _sequence_url_from_sequence_stem(sequence_stem):
    n = _extract_number_after_freiburg(sequence_stem)
    return f"https://cvg.cit.tum.de/rgbd/dataset/freiburg{n}/{sequence_stem}.tgz"
