# SAMExporter — SAM / SAM2 / SAM2.1 / SAM3 / MobileSAM → ONNX

Export [Segment Anything](https://github.com/facebookresearch/segment-anything), [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [Segment Anything 2 / 2.1](https://github.com/facebookresearch/segment-anything-2), and [Segment Anything 3](https://github.com/facebookresearch/sam3) to ONNX for easy, dependency-free deployment.

[![PyPI version](https://badge.fury.io/py/samexporter.svg)](https://badge.fury.io/py/samexporter)
[![Downloads](https://pepy.tech/badge/samexporter)](https://pepy.tech/project/samexporter)
[![Downloads](https://pepy.tech/badge/samexporter/month)](https://pepy.tech/project/samexporter)
[![Downloads](https://pepy.tech/badge/samexporter/week)](https://pepy.tech/project/samexporter)

**Supported models:**

| Model | Prompt types | Notes |
|-------|-------------|-------|
| SAM ViT-B / ViT-L / ViT-H | Point, Rectangle | Original Meta SAM |
| SAM ViT-B / ViT-L / ViT-H (quantized) | Point, Rectangle | Smaller, faster variants |
| MobileSAM | Point, Rectangle | Lightweight; fast on CPU |
| SAM2 Tiny / Small / Base+ / Large | Point, Rectangle | Meta SAM 2 |
| SAM2.1 Tiny / Small / Base+ / Large | Point, Rectangle | Improved SAM 2 |
| SAM3 ViT-H | **Text**, Point, Rectangle | Open-vocabulary text-driven segmentation |

---

## Installation

Requires **Python 3.11+**.

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install samexporter
```

> **Note — Windows users:** The optional `onnxsim` model simplifier (used during ONNX export) has no pre-built wheel for Windows. If you plan to export models and want simplification, install with:
> ```bash
> pip install "samexporter[export]"
> ```
> or enable [Windows Long Path support](https://pip.pypa.io/warnings/enable-long-paths) before installing.

### From source

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
git clone --recurse-submodules https://github.com/vietanhdev/samexporter
cd samexporter
pip install -e .
```

---

## SAM / MobileSAM — Convert to ONNX

### 1. Download checkpoints

Place checkpoints in `original_models/`:

```text
original_models/
  sam_vit_b_01ec64.pth
  sam_vit_l_0b3195.pth
  sam_vit_h_4b8939.pth
  mobile_sam.pt
```

Download links:
- [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt)

### 2. Export encoder

```bash
# SAM ViT-H (most accurate)
python -m samexporter.export_encoder \
    --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.encoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.encoder.quant.onnx \
    --use-preprocess

# SAM ViT-B (fastest)
python -m samexporter.export_encoder \
    --checkpoint original_models/sam_vit_b_01ec64.pth \
    --output output_models/sam_vit_b_01ec64.encoder.onnx \
    --model-type vit_b \
    --quantize-out output_models/sam_vit_b_01ec64.encoder.quant.onnx \
    --use-preprocess
```

### 3. Export decoder

```bash
python -m samexporter.export_decoder \
    --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.decoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.decoder.quant.onnx \
    --return-single-mask
```

Remove `--return-single-mask` to return multiple mask proposals.

**Batch convert all SAM models:**

```bash
bash convert_all_meta_sam.sh
bash convert_mobile_sam.sh
```

### 4. Run inference

```bash
python -m samexporter.inference \
    --encoder_model output_models/sam_vit_h_4b8939.encoder.onnx \
    --decoder_model output_models/sam_vit_h_4b8939.decoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --output output_images/truck.png \
    --show
```

![truck](https://raw.githubusercontent.com/vietanhdev/samexporter/main/sample_outputs/truck.png)

```bash
python -m samexporter.inference \
    --encoder_model output_models/sam_vit_h_4b8939.encoder.onnx \
    --decoder_model output_models/sam_vit_h_4b8939.decoder.onnx \
    --image images/plants.png \
    --prompt images/plants_prompt1.json \
    --output output_images/plants_01.png \
    --show
```

![plants_01](https://raw.githubusercontent.com/vietanhdev/samexporter/main/sample_outputs/plants_01.png)

---

## SAM2 / SAM2.1 — Convert to ONNX

### 1. Download checkpoints

```bash
cd original_models && bash download_sam2.sh
```

Or download manually:

```text
original_models/
  sam2_hiera_tiny.pt
  sam2_hiera_small.pt
  sam2_hiera_base_plus.pt
  sam2_hiera_large.pt
  sam2.1_hiera_tiny.pt
  sam2.1_hiera_small.pt
  sam2.1_hiera_base_plus.pt
  sam2.1_hiera_large.pt
```

### 2. Install SAM2 PyTorch package

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### 3. Export

```bash
# Single model example (SAM2 Tiny)
python -m samexporter.export_sam2 \
    --checkpoint original_models/sam2_hiera_tiny.pt \
    --output_encoder output_models/sam2_hiera_tiny.encoder.onnx \
    --output_decoder output_models/sam2_hiera_tiny.decoder.onnx \
    --model_type sam2_hiera_tiny

# SAM2.1 example
python -m samexporter.export_sam2 \
    --checkpoint original_models/sam2.1_hiera_tiny.pt \
    --output_encoder output_models/sam2.1_hiera_tiny.encoder.onnx \
    --output_decoder output_models/sam2.1_hiera_tiny.decoder.onnx \
    --model_type sam2.1_hiera_tiny
```

**Batch convert all SAM2 / SAM2.1 models:**

```bash
bash convert_all_meta_sam2.sh
```

### 4. Run inference

```bash
python -m samexporter.inference \
    --encoder_model output_models/sam2_hiera_tiny.encoder.onnx \
    --decoder_model output_models/sam2_hiera_tiny.decoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_prompt.json \
    --sam_variant sam2 \
    --output output_images/sam2_truck.png \
    --show
```

![truck_sam2](https://raw.githubusercontent.com/vietanhdev/samexporter/main/sample_outputs/sam2_truck.png)

---

## SAM3 — Convert to ONNX

SAM3 extends the SAM family with open-vocabulary, text-driven segmentation. In addition to point and rectangle prompts, it accepts **text prompts** (e.g., "truck", "person") to detect and segment objects without any prior training on those classes.

SAM3 exports into **three separate ONNX models**: an image encoder, a language (text) encoder, and a decoder.

### Pre-exported ONNX models

Pre-exported models are available on HuggingFace and are downloaded automatically:

```
vietanhdev/segment-anything-3-onnx-models
  sam3_image_encoder.onnx  (+ .data)
  sam3_language_encoder.onnx  (+ .data)
  sam3_decoder.onnx  (+ .data)
```

### Export from PyTorch (optional)

```bash
# Clone the SAM3 source (required for export only, not inference)
git submodule update --init sam3

# Install SAM3 dependencies
pip install osam

# Export (add --simplify for ONNX simplification, requires [export] extra on Windows)
python -m samexporter.export_sam3 \
    --output_dir output_models/sam3 \
    --opset 18
```

### Run inference

**Text-only prompt** (detects all instances matching the text):

```bash
python -m samexporter.inference \
    --sam_variant sam3 \
    --encoder_model output_models/sam3/sam3_image_encoder.onnx \
    --decoder_model output_models/sam3/sam3_decoder.onnx \
    --language_encoder_model output_models/sam3/sam3_language_encoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_sam3.json \
    --text_prompt "truck" \
    --output output_images/truck_sam3.png \
    --show
```

**Text + rectangle prompt** (text guides detection, rectangle refines region):

```bash
python -m samexporter.inference \
    --sam_variant sam3 \
    --encoder_model output_models/sam3/sam3_image_encoder.onnx \
    --decoder_model output_models/sam3/sam3_decoder.onnx \
    --language_encoder_model output_models/sam3/sam3_language_encoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_sam3_box.json \
    --text_prompt "truck" \
    --output output_images/truck_sam3_box.png \
    --show
```

**Text + point prompt:**

```bash
python -m samexporter.inference \
    --sam_variant sam3 \
    --encoder_model output_models/sam3/sam3_image_encoder.onnx \
    --decoder_model output_models/sam3/sam3_decoder.onnx \
    --language_encoder_model output_models/sam3/sam3_language_encoder.onnx \
    --image images/truck.jpg \
    --prompt images/truck_sam3_point.json \
    --text_prompt "truck" \
    --output output_images/truck_sam3_point.png \
    --show
```

> **Note:** Always pass `--text_prompt` for SAM3. Without it the model defaults to a "visual" text token and may produce zero detections.

---

## Prompt JSON format

Prompts are JSON files containing a list of mark objects:

```json
[
  {"type": "point",     "data": [x, y],           "label": 1},
  {"type": "rectangle", "data": [x1, y1, x2, y2]},
  {"type": "text",      "data": "object description"}
]
```

- `label: 1` — foreground point; `label: 0` — background point
- `type: "text"` is specific to SAM3 (use `--text_prompt` on the CLI instead for convenience)

---

## Tips

- Use **quantized** models (`*.quant.onnx`) for faster inference and smaller file size with minimal accuracy loss.
- **SAM ViT-B** is the fastest SAM1 variant; **SAM ViT-H** is the most accurate.
- **SAM2 Tiny / SAM2.1 Tiny** are good CPU-friendly choices for SAM2.
- **SAM3** is slower due to its three-model pipeline but uniquely supports natural-language object queries.
- Run the encoder once per image; the lightweight decoder handles prompt changes in real time.

---

## Running tests

```bash
pip install pytest
pytest tests/
```

---

## AnyLabeling

This package was originally developed for the auto-labeling feature in [AnyLabeling](https://github.com/vietanhdev/anylabeling). However, it can be used independently for any ONNX-based deployment scenario.

[![AnyLabeling](https://user-images.githubusercontent.com/18329471/236625792-07f01838-3f69-48b0-a12e-30bad27bd921.gif)](https://youtu.be/5qVJiYNX5Kk)

---

## License

MIT — see [LICENSE](LICENSE) for details.

## References

- ONNX-SAM2-Segment-Anything: [https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)
- sam3-onnx: [https://github.com/wkentaro/sam3-onnx](https://github.com/wkentaro/sam3-onnx)
