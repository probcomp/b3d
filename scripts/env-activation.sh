#!/usr/bin/env bash

set -euo pipefail

capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

export TORCH_CUDA_ARCH_LIST="$capability"
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

if [ -n "${CONDA_PREFIX:-}" ]; then
    export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"
    LIB_DIR="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"
    if [ -d "$LIB_DIR" ] && [ ! -e "$LIB_DIR/libEGL.so" ]; then
        cp assets/system/libEGL.so "$LIB_DIR/"
    fi
fi
