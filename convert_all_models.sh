#!/bin/bash
# convert_all_models.sh

set -e

export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p output_models/mobile_sam
mkdir -p output_models/sam3

echo "=== Converting Segment Anything (SAM 1) ==="

echo "Converting ViT-H models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_h_4b8939.pth --output output_models/sam_vit_h_4b8939.encoder.onnx --model-type vit_h --quantize-out output_models/sam_vit_h_4b8939.encoder.quant.onnx --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_h_4b8939.pth --output output_models/sam_vit_h_4b8939.decoder.onnx --model-type vit_h --quantize-out output_models/sam_vit_h_4b8939.decoder.quant.onnx --return-single-mask

echo "Converting ViT-L models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_l_0b3195.pth --output output_models/sam_vit_l_0b3195.encoder.onnx --model-type vit_l --quantize-out output_models/sam_vit_l_0b3195.encoder.quant.onnx --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_l_0b3195.pth --output output_models/sam_vit_l_0b3195.decoder.onnx --model-type vit_l --quantize-out output_models/sam_vit_l_0b3195.decoder.quant.onnx --return-single-mask

echo "Converting ViT-B models..."
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_b_01ec64.pth --output output_models/sam_vit_b_01ec64.encoder.onnx --model-type vit_b --quantize-out output_models/sam_vit_b_01ec64.encoder.quant.onnx --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_b_01ec64.pth --output output_models/sam_vit_b_01ec64.decoder.onnx --model-type vit_b --quantize-out output_models/sam_vit_b_01ec64.decoder.quant.onnx --return-single-mask

echo -e "\n=== Converting Mobile SAM ==="
python -m samexporter.export_encoder --checkpoint original_models/mobile_sam.pt --output output_models/mobile_sam/mobile_sam.encoder.onnx --model-type mobile --quantize-out output_models/mobile_sam/mobile_sam.encoder.quant.onnx --use-preprocess
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_h_4b8939.pth --output output_models/mobile_sam/sam_vit_h_4b8939.decoder.onnx --model-type vit_h --quantize-out output_models/mobile_sam/sam_vit_h_4b8939.decoder.quant.onnx --return-single-mask

echo -e "\n=== Converting Segment Anything 2 (SAM 2) ==="

echo "Converting SAM2-Hiera-Tiny models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2_hiera_tiny.pt --output_encoder output_models/sam2_hiera_tiny.encoder.onnx --output_decoder output_models/sam2_hiera_tiny.decoder.onnx --model_type sam2_hiera_tiny

echo "Converting SAM2-Hiera-Small models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2_hiera_small.pt --output_encoder output_models/sam2_hiera_small.encoder.onnx --output_decoder output_models/sam2_hiera_small.decoder.onnx --model_type sam2_hiera_small

echo "Converting SAM2-Hiera-BasePlus models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2_hiera_base_plus.pt --output_encoder output_models/sam2_hiera_base_plus.encoder.onnx --output_decoder output_models/sam2_hiera_base_plus.decoder.onnx --model_type sam2_hiera_base_plus

echo "Converting SAM2-Hiera-Large models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2_hiera_large.pt --output_encoder output_models/sam2_hiera_large.encoder.onnx --output_decoder output_models/sam2_hiera_large.decoder.onnx --model_type sam2_hiera_large

echo -e "\n=== Converting Segment Anything 2.1 (SAM 2.1) ==="

echo "Converting SAM2.1-Hiera-Tiny models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2.1_hiera_tiny.pt --output_encoder output_models/sam2.1_hiera_tiny.encoder.onnx --output_decoder output_models/sam2.1_hiera_tiny.decoder.onnx --model_type sam2.1_hiera_tiny

echo "Converting SAM2.1-Hiera-Small models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2.1_hiera_small.pt --output_encoder output_models/sam2.1_hiera_small.encoder.onnx --output_decoder output_models/sam2.1_hiera_small.decoder.onnx --model_type sam2.1_hiera_small

echo "Converting SAM2.1-Hiera-BasePlus models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2.1_hiera_base_plus.pt --output_encoder output_models/sam2.1_hiera_base_plus.encoder.onnx --output_decoder output_models/sam2.1_hiera_base_plus.decoder.onnx --model_type sam2.1_hiera_base_plus

echo "Converting SAM2.1-Hiera-Large models..."
python -m samexporter.export_sam2 --checkpoint original_models/sam2.1_hiera_large.pt --output_encoder output_models/sam2.1_hiera_large.encoder.onnx --output_decoder output_models/sam2.1_hiera_large.decoder.onnx --model_type sam2.1_hiera_large

echo -e "\n=== Converting Segment Anything 3 (SAM 3) ==="
python -m samexporter.export_sam3 --output_dir output_models/sam3

echo -e "\nAll conversions complete!"
