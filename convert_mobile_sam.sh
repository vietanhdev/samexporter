#!/usr/bin/env bash

set -euo pipefail

echo "Converting Mobile SAM..."
python -m samexporter.export_encoder --checkpoint original_models/mobile_sam.pt \
    --output output_models/mobile_sam/mobile_sam.encoder.onnx \
    --model-type mobile \
    --quantize-out output_models/mobile_sam/mobile_sam.encoder.quant.onnx \
    --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/mobile_sam.pt \
    --output output_models/mobile_sam/mobile_sam.decoder.onnx \
    --model-type mobile \
    --quantize-out output_models/mobile_sam/mobile_sam.decoder.quant.onnx \
    --return-single-mask
