# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import setuptools

setuptools.setup(
    name="b3d",
    version="0.0.1",
    packages=setuptools.find_packages(),
    package_data={
        "b3d": [
            "nvdiffrast/common/*.h",
            "nvdiffrast/common/*.inl",
            "nvdiffrast/common/*.cu",
            "nvdiffrast/common/*.cpp",
            "nvdiffrast/common/cudaraster/*.hpp",
            "nvdiffrast/common/cudaraster/impl/*.cpp",
            "nvdiffrast/common/cudaraster/impl/*.hpp",
            "nvdiffrast/common/cudaraster/impl/*.inl",
            "nvdiffrast/common/cudaraster/impl/*.cu",
            "nvdiffrast/lib/*.h",
            "nvdiffrast/torch/*.h",
            "nvdiffrast/torch/*.inl",
            "nvdiffrast/torch/*.cpp",
            "nvdiffrast/tensorflow/*.cu",
            "nvdiffrast/jax/*.h",
            "nvdiffrast/jax/*.inl",
            "nvdiffrast/jax/*.cpp",
        ]
        + (["lib/*.lib"] if os.name == "nt" else [])
    },
    include_package_data=True,
    dependency_links=[
        "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    ],
    install_requires=[
        "genjax @ git+https://github.com/probcomp/genjax",
        "rerun-sdk==0.14.1",
        "tqdm==4.66.2",
        "numpy==1.26.4",
        "pillow==10.3.0",
        "tensorflow-probability==0.23.0",
        "trimesh==4.2.4",
        "matplotlib==3.8.4",
        "scipy==1.13.0",
        "ninja==1.11.1.1",
        "scikit-learn==1.4.1.post1",
        "pytest==8.1.1",
        "ipython==8.23.0",
        "jupyter==1.0.0",
        "pyransac3d==0.6.0",
        "pdoc3==0.10.0",
        "opencv-python==4.9.0.80",
        "fire==0.6.0",
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "torchvision==0.17.2",
        "jax[cuda12_pip]==0.4.26",
        "natsort",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "b3d_pull = b3d.bucket_utils.b3d_pull:download_from_bucket",
            "b3d_push = b3d.bucket_utils.b3d_push:upload_to_bucket",
        ]
    },
    python_requires=">=3.6",
)
