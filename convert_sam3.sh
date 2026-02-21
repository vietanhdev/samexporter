#!/bin/bash
# Export SAM3 ViT-H to ONNX.
#
# Requirements:
#   - sam3 submodule initialised: git submodule update --init sam3
#   - osam installed for CLIP tokenisation: pip install osam
#
# Optional: pass --simplify to run onnxsim after export (reduces some
# redundant ops; vision_pos_enc_0/1 may be removed from the decoder).

set -euo pipefail

OUTPUT_DIR="${1:-output_models/sam3}"
SIMPLIFY="${SIMPLIFY:-}"

echo "Exporting SAM3 ViT-H to ONNX → $OUTPUT_DIR"

if [ -n "$SIMPLIFY" ]; then
    python -m samexporter.export_sam3 \
        --output_dir "$OUTPUT_DIR" \
        --opset 18 \
        --simplify
else
    python -m samexporter.export_sam3 \
        --output_dir "$OUTPUT_DIR" \
        --opset 18
fi

echo "Done – models written to $OUTPUT_DIR/"
echo "  sam3_image_encoder.onnx"
echo "  sam3_language_encoder.onnx"
echo "  sam3_decoder.onnx"
