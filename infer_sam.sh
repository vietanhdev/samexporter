python -m samexporter.inference \
    --encoder_model output_models/sam_vit_b_01ec64.encoder.quant.onnx \
    --decoder_model output_models/sam_vit_b_01ec64.decoder.quant.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --output output_images/truck.jpg \
    --show
