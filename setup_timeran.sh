#!/bin/bash
# Setup script for TimeRAN (clones the official repo and checks checkpoints)
# Run this from the repository root

set -e

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${VENV:-$PROJ_DIR/venv/bin}"

echo "=== Installing momentfm and peft in existing venv ==="
$VENV/pip3 install momentfm peft gdown tqdm --quiet

echo "=== Cloning TimeRAN repository ==="
cd "$PROJ_DIR"
if [ ! -d "TimeRAN" ]; then
    git clone https://github.com/panitsasi/TimeRAN.git
else
    echo "TimeRAN repo already exists, pulling latest..."
    cd TimeRAN && git pull && cd ..
fi

echo "=== Downloading TimeRAN checkpoints ==="
CKPT_DIR="$PROJ_DIR/TimeRAN/TimeRAN_Foundation_Model/data/checkpoints"
mkdir -p "$CKPT_DIR/small" "$CKPT_DIR/base" "$CKPT_DIR/large"

# Checkpoint Google Drive IDs (from TimeRAN README)
# base:  1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ
# small: 1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ
# large: 1We9zE5BV6Iwkc_EKSAhP28B3wcM7RZRd

if [ ! -f "$CKPT_DIR/base/TimeRAN_base.pth" ]; then
    echo "Downloading TimeRAN-base checkpoint..."
    $VENV/gdown "1fJNCkufmfWC6zHecz10PUyreD0PhBOMJ" -O "$CKPT_DIR/base/TimeRAN_base.pth"
else
    echo "TimeRAN-base checkpoint already exists."
fi

if [ ! -f "$CKPT_DIR/small/TimeRAN_small.pth" ]; then
    echo "Downloading TimeRAN-small checkpoint..."
    $VENV/gdown "1gz23mmP4ZiNznCloObEaSlVaJH21fyxJ" -O "$CKPT_DIR/small/TimeRAN_small.pth"
else
    echo "TimeRAN-small checkpoint already exists."
fi

echo "=== Setup complete ==="
echo "Run the evaluation with:"
echo "  $VENV/python3 $PROJ_DIR/run_timeran.py --dataset gotsf"
echo "  $VENV/python3 $PROJ_DIR/run_timeran.py --dataset irish"
