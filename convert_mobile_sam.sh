echo "Converting Mobile SAM..."
python -m samexporter.export_encoder --checkpoint original_models/mobile_sam.pt \
    --output output_models/mobile_sam/mobile_sam.encoder.onnx \
    --model-type mobile \
    --quantize-out output_models/mobile_sam/mobile_sam.encoder.quant.onnx \
    --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/mobile_sam/sam_vit_h_4b8939.decoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/mobile_sam/sam_vit_h_4b8939.decoder.quant.onnx \
    --return-single-mask
