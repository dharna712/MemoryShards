#!/usr/bin/env bash
# Render build command. facenet-pytorch==2.6.0's published metadata pins
# numpy<2.0.0, Pillow<10.3.0, torch<2.3.0, torchvision<0.18.0 — badly
# outdated ranges that conflict with what transformers/datasets need.
# Its actual runtime code works fine against modern versions; only its
# declared metadata doesn't. Installed separately with --no-deps so pip's
# resolver never sees those stale constraints — same as the local setup.
set -e
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
