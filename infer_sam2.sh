python -m samexporter.inference \
    --encoder_model output_models/sam2_hiera_tiny.encoder.onnx \
    --decoder_model output_models/sam2_hiera_tiny.decoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --output output_images/sam2_truck.png \
    --sam_variant sam2 \
    --show
