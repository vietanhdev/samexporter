#!/bin/bash
# SAM3 inference examples.
#
# SAM3 supports three prompt modes:
#   1. Text only        – open-vocabulary detection (no geometric hint needed)
#   2. Text + point     – refine detection around a clicked pixel
#   3. Text + rectangle – constrain detection to a bounding box
#
# The --text_prompt flag drives the language encoder; always supply it for
# best results.  If omitted the model falls back to "visual" (no language
# guidance) which may return fewer or no detections.

set -euo pipefail

ENC="output_models/sam3/sam3_image_encoder.onnx"
DEC="output_models/sam3/sam3_decoder.onnx"
LANG="output_models/sam3/sam3_language_encoder.onnx"
IMG="images/truck.jpg"

echo "--- SAM3: text-only prompt ('truck') ---"
python -m samexporter.inference \
    --encoder_model "$ENC" \
    --decoder_model "$DEC" \
    --language_encoder_model "$LANG" \
    --image "$IMG" \
    --prompt images/truck_sam3.json \
    --text_prompt "truck" \
    --output output_images/sam3_truck_text.png \
    --sam_variant sam3

echo "--- SAM3: text + rectangle prompt ---"
python -m samexporter.inference \
    --encoder_model "$ENC" \
    --decoder_model "$DEC" \
    --language_encoder_model "$LANG" \
    --image "$IMG" \
    --prompt images/truck_sam3_box.json \
    --text_prompt "truck" \
    --output output_images/sam3_truck_box.png \
    --sam_variant sam3

echo "--- SAM3: text + point prompt ---"
python -m samexporter.inference \
    --encoder_model "$ENC" \
    --decoder_model "$DEC" \
    --language_encoder_model "$LANG" \
    --image "$IMG" \
    --prompt images/truck_sam3_point.json \
    --text_prompt "truck" \
    --output output_images/sam3_truck_point.png \
    --sam_variant sam3

echo "Done – outputs saved to output_images/"
