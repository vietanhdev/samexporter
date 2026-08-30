#!/bin/bash
# Real SAM3 text, geometric-selection, and capped-discovery examples.
# Geometry is a concept exemplar. The default auto mode returns the best mask
# overlapping a point/rectangle; pass --sam3_output_mode all to keep every
# visually matching instance.

set -euo pipefail

ENC="output_models/sam3/sam3_image_encoder.onnx"
DEC="output_models/sam3/sam3_decoder.onnx"
LANG="output_models/sam3/sam3_language_encoder.onnx"
OUT_DIR="${SAMEXPORTER_RESULTS_DIR:-visual_results/runs/manual/sam3}"
mkdir -p "$OUT_DIR"

run_case() {
    local name=$1
    local image=$2
    local prompt=$3
    shift 3
    python -m samexporter.inference \
        --sam_variant sam3 \
        --encoder_model "$ENC" \
        --decoder_model "$DEC" \
        --language_encoder_model "$LANG" \
        --image "$image" \
        --prompt "$prompt" \
        --output "$OUT_DIR/$name.png" \
        "$@" 2>&1 | tee "$OUT_DIR/$name.log"
}

# A singular text prompt discovers the truck.
run_case truck_text images/truck.jpg images/truck_sam3.json \
    --text_prompt "truck"

# Text plus geometry reliably chooses the intended matching instance.
run_case truck_text_box images/truck.jpg images/truck_sam3_box.json \
    --text_prompt "truck"
run_case truck_text_point images/truck.jpg images/truck_sam3_point.json \
    --text_prompt "truck"

# A broad concept discovers every visible plant; cap it for previews.
run_case plants_text_all images/plants.png images/plants_text.json \
    --text_prompt "plant"
run_case plants_text_top5 images/plants.png images/plants_text.json \
    --text_prompt "plant" --max_instances 5

# Geometry-only auto mode uses the generic visual token, then returns the best
# prompt-overlapping match instead of every similar object in the image.
run_case plants_box images/plants.png images/plants_box.json
run_case plants_box_refined images/plants.png images/plants_box_refined.json

echo "SAM3 images and sibling logs saved to $OUT_DIR"
