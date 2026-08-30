#!/bin/bash
set -euo pipefail

OUT_DIR="${SAMEXPORTER_RESULTS_DIR:-visual_results/runs/manual}"
mkdir -p "$OUT_DIR"

python -m samexporter.inference \
    --encoder_model output_models/sam2_hiera_tiny.encoder.onnx \
    --decoder_model output_models/sam2_hiera_tiny.decoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --output "$OUT_DIR/sam2_truck.png" \
    --sam_variant sam2 \
    --show 2>&1 | tee "$OUT_DIR/sam2_truck.log"
