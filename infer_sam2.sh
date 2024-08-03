python -m samexporter.inference \
    --encoder_model output_models/sam2_hiera_tiny.encoder.onnx \
    --decoder_model output_models/sam2_hiera_tiny.decoder.onnx \
    --image images/plants.png \
    --prompt images/truck_prompt_2.json \
    --output output_images/plants_prompt_2_sam2.png \
    --sam_variant sam2 \
    --show
