python -m samexporter.inference \
    --encoder_model output_models/mobile_sam/mobile_sam.encoder.onnx \
    --decoder_model output_models/mobile_sam/sam_vit_h_4b8939.decoder.onnx \
    --image images/plants.png \
    --prompt images/plants_prompt1.json \
    --output output_images/plants_01.png \
    --show
