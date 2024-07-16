#!/usr/bin/env bash

set -euo pipefail

capability=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)

export TORCH_CUDA_ARCH_LIST="$capability"
