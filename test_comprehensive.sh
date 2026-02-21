#!/bin/bash
# test_comprehensive.sh

set -e

OUT_DIR="test_outputs/comprehensive"
mkdir -p "$OUT_DIR"

# Parameters: variant, encoder, decoder, image, prompt, suffix, extra_args
run_single_test() {
    local variant=$1
    local encoder=$2
    local decoder=$3
    local image=$4
    local prompt=$5
    local suffix=$6
    local extra_args=$7
    
    local img_base=$(basename "$image" | cut -d. -f1)
    local enc_base=$(basename "$encoder" | cut -d. -f1)
    local output="${OUT_DIR}/${variant}_${enc_base}_${img_base}_${suffix}.png"

    echo "Testing: $variant | Model: $enc_base | Image: $img_base | Mode: $suffix"
    
    python -m samexporter.inference \
        --sam_variant "$variant" \
        --encoder_model "$encoder" \
        --decoder_model "$decoder" \
        --image "$image" \
        --prompt "$prompt" \
        --output "$output" \
        $extra_args > /dev/null 2>&1
    
    if [ -f "$output" ]; then
        echo "  [OK] -> $output"
    else
        echo "  [FAIL] -> $output"
        exit 1
    fi
}

MODELS_SAM1=("sam_vit_h_4b8939")
MODELS_SAM2=("sam2_hiera_tiny" "sam2_hiera_large")
MODELS_SAM21=("sam2.1_hiera_tiny" "sam2.1_hiera_large")

IMAGES=("images/truck.jpg" "images/plants.png")

echo "=== Starting Comprehensive Tests ==="

# 1. Test SAM 1 & Mobile SAM
for img in "${IMAGES[@]}"; do
    img_name=$(basename "$img" | cut -d. -f1)
    # Point
    run_single_test "sam" "output_models/sam_vit_h_4b8939.encoder.onnx" "output_models/sam_vit_h_4b8939.decoder.onnx" "$img" "images/${img_name}_point.json" "point"
    # Box
    run_single_test "sam" "output_models/sam_vit_h_4b8939.encoder.onnx" "output_models/sam_vit_h_4b8939.decoder.onnx" "$img" "images/${img_name}_box.json" "box"
    # Mobile SAM
    run_single_test "sam" "output_models/mobile_sam/mobile_sam.encoder.onnx" "output_models/mobile_sam/sam_vit_h_4b8939.decoder.onnx" "$img" "images/${img_name}_box.json" "mobile_box"
done

# 2. Test SAM 2 & 2.1
for model in "${MODELS_SAM2[@]}" "${MODELS_SAM21[@]}"; do
    for img in "${IMAGES[@]}"; do
        img_name=$(basename "$img" | cut -d. -f1)
        run_single_test "sam2" "output_models/${model}.encoder.onnx" "output_models/${model}.decoder.onnx" "$img" "images/${img_name}_point.json" "point"
        run_single_test "sam2" "output_models/${model}.encoder.onnx" "output_models/${model}.decoder.onnx" "$img" "images/${img_name}_box.json" "box"
    done
done

# 3. Test SAM 3 (Point, Box, Text)
for img in "${IMAGES[@]}"; do
    img_name=$(basename "$img" | cut -d. -f1)
    # Point
    run_single_test "sam3" "output_models/sam3/sam3_image_encoder.onnx" "output_models/sam3/sam3_decoder.onnx" "$img" "images/${img_name}_point.json" "point" "--language_encoder_model output_models/sam3/sam3_language_encoder.onnx"
    # Box
    run_single_test "sam3" "output_models/sam3/sam3_image_encoder.onnx" "output_models/sam3/sam3_decoder.onnx" "$img" "images/${img_name}_box.json" "box" "--language_encoder_model output_models/sam3/sam3_language_encoder.onnx"
done

# SAM 3 specific Text tests
run_single_test "sam3" "output_models/sam3/sam3_image_encoder.onnx" "output_models/sam3/sam3_decoder.onnx" "images/truck.jpg" "images/truck_sam3.json" "text_truck" "--language_encoder_model output_models/sam3/sam3_language_encoder.onnx"
run_single_test "sam3" "output_models/sam3/sam3_image_encoder.onnx" "output_models/sam3/sam3_decoder.onnx" "images/plants.png" "images/plants_text.json" "text_leaf" "--language_encoder_model output_models/sam3/sam3_language_encoder.onnx"

echo -e "
All comprehensive tests passed!"
