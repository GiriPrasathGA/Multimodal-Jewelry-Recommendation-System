#!/usr/bin/env bash
# Exit on error
set -o errexit

pip install --upgrade pip
# Install CPU-only torch to save space (approx 700MB -> 150MB)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
