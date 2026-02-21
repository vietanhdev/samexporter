#!/bin/bash
# download_all_models.sh

set -e

OUT_DIR="original_models"
mkdir -p "$OUT_DIR"

download_file() {
    local url=$1
    local dest=$2
    if [ -f "$dest" ]; then
        echo "  [SKIP] Already exists: $dest"
    else
        echo "  Downloading $dest ..."
        curl -L "$url" -o "$dest"
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
# SAM3 is a zip file containing multiple components
if [ -f "output_models/sam3/sam3_image_encoder.onnx" ]; then
    echo "  [SKIP] SAM3 already exported or present in output_models/sam3"
else
    # We download the zip if not already present
    download_file "https://huggingface.co/vietanhdev/segment-anything-3-onnx-models/resolve/main/sam3_vit_h.zip" "$OUT_DIR/sam3_vit_h.zip"
    echo "  Extracting SAM3 models..."
    mkdir -p output_models/sam3
    unzip -o "$OUT_DIR/sam3_vit_h.zip" -d output_models/sam3
fi

echo -e "
All downloads complete!"
ls -lh "$OUT_DIR"
