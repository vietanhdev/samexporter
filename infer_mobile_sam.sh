#!/bin/bash
set -euo pipefail

OUT_DIR="${SAMEXPORTER_RESULTS_DIR:-visual_results/runs/manual}"
mkdir -p "$OUT_DIR"
python -m samexporter.inference \
    --encoder_model output_models/mobile_sam/mobile_sam.encoder.onnx \
    --decoder_model output_models/mobile_sam/mobile_sam.decoder.onnx \
    --image images/plants.png \
    --prompt images/plants_prompt1.json \
    --output "$OUT_DIR/mobile_sam_plants_01.png" \
    --show 2>&1 | tee "$OUT_DIR/mobile_sam_plants_01.log"
