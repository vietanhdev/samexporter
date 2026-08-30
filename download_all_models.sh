#!/bin/bash
# download_all_models.sh

set -euo pipefail

OUT_DIR="original_models"
mkdir -p "$OUT_DIR"

download_file() {
    local url=$1
    local dest=$2
    if [ -f "$dest" ]; then
        echo "  [SKIP] Already exists: $dest"
    else
        echo "  Downloading $dest ..."
        mkdir -p "$(dirname "$dest")"
        curl --fail --location --retry 3 --continue-at - \
            "$url" --output "$dest.part"
        mv "$dest.part" "$dest"
        echo "  [OK] $dest"
    fi
}

echo -e "
=== Segment Anything (SAM 1) ==="
download_file "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" "$OUT_DIR/sam_vit_h_4b8939.pth"
download_file "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" "$OUT_DIR/sam_vit_l_0b3195.pth"
download_file "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" "$OUT_DIR/sam_vit_b_01ec64.pth"

echo -e "
=== MobileSAM ==="
download_file "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt" "$OUT_DIR/mobile_sam.pt"

echo -e "
=== EfficientSAM-Ti (ONNX) ==="
mkdir -p output_models/efficient_sam
download_file "https://huggingface.co/nrl-ai/samexporter-onnx-models/resolve/main/efficient_sam_ti/efficientsam_ti_encoder.onnx" "output_models/efficient_sam/efficientsam_ti_encoder.onnx"
download_file "https://huggingface.co/nrl-ai/samexporter-onnx-models/resolve/main/efficient_sam_ti/efficientsam_ti_decoder.onnx" "output_models/efficient_sam/efficientsam_ti_decoder.onnx"

echo -e "
=== Segment Anything 2 (SAM 2) ==="
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" "$OUT_DIR/sam2_hiera_tiny.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt" "$OUT_DIR/sam2_hiera_small.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt" "$OUT_DIR/sam2_hiera_base_plus.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt" "$OUT_DIR/sam2_hiera_large.pt"

echo -e "\n=== Segment Anything 2.1 (SAM 2.1) ==="
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" "$OUT_DIR/sam2.1_hiera_tiny.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" "$OUT_DIR/sam2.1_hiera_small.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" "$OUT_DIR/sam2.1_hiera_base_plus.pt"
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" "$OUT_DIR/sam2.1_hiera_large.pt"

echo -e "\n=== Segment Anything 3 (SAM 3) ==="
# SAM3 uses one external-data companion file for its decoder. Check every file
# independently so an interrupted or partial prior download is repaired.
mkdir -p output_models/sam3
SAMEXPORTER_HF="https://huggingface.co/nrl-ai/samexporter-onnx-models/resolve/main/sam3"
download_file "$SAMEXPORTER_HF/sam3_image_encoder.onnx" "output_models/sam3/sam3_image_encoder.onnx"
download_file "$SAMEXPORTER_HF/sam3_language_encoder.onnx" "output_models/sam3/sam3_language_encoder.onnx"
download_file "$SAMEXPORTER_HF/sam3_decoder.onnx" "output_models/sam3/sam3_decoder.onnx"
download_file "$SAMEXPORTER_HF/sam3_decoder.onnx.data" "output_models/sam3/sam3_decoder.onnx.data"

echo -e "
All downloads complete!"
ls -lh "$OUT_DIR"
