#!/bin/bash
conda activate b3d
sudo apt-get install mesa-common-dev libegl1-mesa-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
sudo apt-get install ffmpeg
pip install -e .
b3d_pull
mkdir -p assets/test_results