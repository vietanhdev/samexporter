echo "Converting all models..."

echo "Converting ViT-H models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.encoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.encoder.quant.onnx \
    --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.decoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.decoder.quant.onnx \
    --return-single-mask

echo "Converting ViT-L models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_l_0b3195.pth \
    --output output_models/sam_vit_l_0b3195.encoder.onnx \
    --model-type vit_l \
    --quantize-out output_models/sam_vit_l_0b3195.encoder.quant.onnx \
    --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_l_0b3195.pth \
    --output output_models/sam_vit_l_0b3195.decoder.onnx \
    --model-type vit_l \
    --quantize-out output_models/sam_vit_l_0b3195.decoder.quant.onnx \
    --return-single-mask

echo "Converting ViT-B models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_b_01ec64.pth \
    --output output_models/sam_vit_b_01ec64.encoder.onnx \
    --model-type vit_b \
    --quantize-out output_models/sam_vit_b_01ec64.encoder.quant.onnx \
    --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_b_01ec64.pth \
    --output output_models/sam_vit_b_01ec64.decoder.onnx \
    --model-type vit_b \
    --quantize-out output_models/sam_vit_b_01ec64.decoder.quant.onnx \
    --return-single-mask
