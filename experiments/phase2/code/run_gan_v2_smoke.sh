#!/usr/bin/env bash
# Light smoke test before queuing full GAN retrain on RunPod.
set -euo pipefail
cd "$(dirname "$0")/../../.."
python smoke_gan_v2.py "$@"
