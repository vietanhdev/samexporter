#!/bin/bash
# test_all_models.sh

set -e

mkdir -p test_outputs

run_test() {
    local variant=$1
    local encoder=$2
    local decoder=$3
    local prompt_file=$4
    local output="test_outputs/${variant}_$(basename $encoder .onnx)_result.png"
    local extra_args=$5

    echo "Testing $variant with $encoder ..."
    python -m samexporter.inference \
        --sam_variant "$variant" \
        --encoder_model "$encoder" \
        --decoder_model "$decoder" \
        --image images/truck.jpg \
        --prompt "$prompt_file" \
        --output "$output" \
        $extra_args
    
    if [ -f "$output" ]; then
        echo "  [OK] Saved to $output"
    else
        echo "  [FAIL] Failed to save $output"
        exit 1
    fi
}

echo "=== Testing SAM 1 ==="
run_test "sam" "output_models/sam_vit_h_4b8939.encoder.onnx" "output_models/sam_vit_h_4b8939.decoder.onnx" "images/truck_prompt.json"
run_test "sam" "output_models/sam_vit_l_0b3195.encoder.onnx" "output_models/sam_vit_l_0b3195.decoder.onnx" "images/truck_prompt.json"
run_test "sam" "output_models/sam_vit_b_01ec64.encoder.onnx" "output_models/sam_vit_b_01ec64.decoder.onnx" "images/truck_prompt.json"

echo -e "\n=== Testing Mobile SAM ==="
run_test "sam" "output_models/mobile_sam/mobile_sam.encoder.onnx" "output_models/mobile_sam/sam_vit_h_4b8939.decoder.onnx" "images/truck_prompt.json"

echo -e "\n=== Testing SAM 2 ==="
run_test "sam2" "output_models/sam2_hiera_tiny.encoder.onnx" "output_models/sam2_hiera_tiny.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2_hiera_small.encoder.onnx" "output_models/sam2_hiera_small.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2_hiera_base_plus.encoder.onnx" "output_models/sam2_hiera_base_plus.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2_hiera_large.encoder.onnx" "output_models/sam2_hiera_large.decoder.onnx" "images/truck_prompt.json"

echo -e "\n=== Testing SAM 2.1 ==="
run_test "sam2" "output_models/sam2.1_hiera_tiny.encoder.onnx" "output_models/sam2.1_hiera_tiny.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2.1_hiera_small.encoder.onnx" "output_models/sam2.1_hiera_small.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2.1_hiera_base_plus.encoder.onnx" "output_models/sam2.1_hiera_base_plus.decoder.onnx" "images/truck_prompt.json"
run_test "sam2" "output_models/sam2.1_hiera_large.encoder.onnx" "output_models/sam2.1_hiera_large.decoder.onnx" "images/truck_prompt.json"

echo -e "\n=== Testing SAM 3 ==="
run_test "sam3" "output_models/sam3/sam3_image_encoder.onnx" "output_models/sam3/sam3_decoder.onnx" "images/truck_sam3_box.json" "--language_encoder_model output_models/sam3/sam3_language_encoder.onnx"

echo -e "\nAll tests completed successfully!"
