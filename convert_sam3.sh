#!/bin/bash
# Export SAM3 ViT-H to ONNX.
#
# Requirements:
#   - sam3 submodule initialised: git submodule update --init sam3
#   - pinned SAM3 dependencies installed: pip install -e sam3
#
# Optional: pass --simplify to run onnxsim after export (reduces some
# redundant ops; vision_pos_enc_0/1 may be removed from the decoder).

set -euo pipefail

OUTPUT_DIR="${1:-output_models/sam3}"
SIMPLIFY="${SIMPLIFY:-}"
DEVICE="${SAMEXPORTER_EXPORT_DEVICE:-auto}"
MAX_PROMPTS="${SAMEXPORTER_MAX_GEOMETRIC_PROMPTS:-8}"

echo "Exporting SAM3 ViT-H to ONNX → $OUTPUT_DIR"

export_args=(
    --output_dir "$OUTPUT_DIR"
    --opset 18
    --device "$DEVICE"
    --max-geometric-prompts "$MAX_PROMPTS"
)
if [ -n "$SIMPLIFY" ]; then
    export_args+=(--simplify)
fi
python -m samexporter.export_sam3 "${export_args[@]}"

echo "Done – models written to $OUTPUT_DIR/"
echo "  sam3_image_encoder.onnx"
echo "  sam3_language_encoder.onnx"
echo "  sam3_decoder.onnx"
