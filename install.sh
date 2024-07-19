#!/bin/bash
conda activate b3d
sudo apt-get install  -y --no-install-recommends \
    ffmpeg \
    mesa-common-dev \
    libegl1-mesa-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev

pip install keyring keyrings.google-artifactregistry-auth
pip install -e . --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
b3d_pull
mkdir -p assets/test_results
