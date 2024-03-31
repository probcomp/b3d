# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import setuptools
import b3d

import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="b3d",
    version=b3d.__version__,
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
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts': [
            'b3d_pull = b3d.bucket_utils.chi_pull:download_from_bucket',
            'b3d_push = b3d.bucket_utils.chi_push:upload_to_bucket',
        ]
        
    },
    python_requires=">=3.6",
)
