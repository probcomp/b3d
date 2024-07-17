#!/usr/bin/env bash

set -euo pipefail

capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

export TORCH_CUDA_ARCH_LIST="$capability"
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include"
