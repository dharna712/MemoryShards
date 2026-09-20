#!/usr/bin/env bash
# Render build command.
set -e

# Plain `pip install torch` pulls PyPI's default CUDA-enabled build
# (nvidia-cublas, cuda-toolkit, triton, etc.) — those CUDA runtime
# libraries get loaded into memory as soon as torch is imported, even on
# a CPU-only box like Render's free tier. Confirmed this was pinning
# memory at the 512MB free-tier ceiling before the server ever finished
# booting. Installing from PyTorch's own CPU-only index avoids all of it.
pip install torch>=2.12 torchvision>=0.27 --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

# facenet-pytorch==2.6.0's published metadata pins numpy<2.0.0,
# Pillow<10.3.0, torch<2.3.0, torchvision<0.18.0 — badly outdated ranges
# that conflict with what transformers/datasets need. Its actual runtime
# code works fine against modern versions; only its declared metadata
# doesn't. Installed separately with --no-deps so pip's resolver never
# sees those stale constraints — same as the local setup.
pip install --no-deps facenet-pytorch==2.6.0
