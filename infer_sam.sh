#!/bin/bash
set -euo pipefail

OUT_DIR="${SAMEXPORTER_RESULTS_DIR:-visual_results/runs/manual}"
mkdir -p "$OUT_DIR"

python -m samexporter.inference \
    --encoder_model output_models/sam_vit_b_01ec64.encoder.quant.onnx \
    --decoder_model output_models/sam_vit_b_01ec64.decoder.quant.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --output "$OUT_DIR/sam_vit_b_truck.png" \
    --show 2>&1 | tee "$OUT_DIR/sam_vit_b_truck.log"
