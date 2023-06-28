# SAM Exporter

Exporting [Segment Anything](https://github.com/facebookresearch/segment-anything) models to different formats.

The [Segment Anything repository](https://github.com/facebookresearch/segment-anything) does not have a way to export **encoder** to ONNX format. There are some pull requests for this feature, but they have not accepted by SAM authors. Therefore, I want to create an easy tool to export Segment Anything models to different output formats as an easy option.

**Supported models:**

- SAM ViT-B
- SAM ViT-L
- SAM ViT-H
- MobileSAM*

## Installation

From PyPi:

```bash
pip install samexporter
```

From source:

```bash
git clone https://github.com/vietanhdev/samexporter
cd samexporter
pip install -e .
```

## Usage

- Download all models from [Segment Anything](https://github.com/facebookresearch/segment-anything) repository (*.pth).
- Download MobileSAM from [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

```text
original_models
   + sam_vit_b_01ec64.pth
   + sam_vit_h_4b8939.pth
   + sam_vit_l_0b3195.pth
   + mobile_sam.pt
   ...
```

- Convert encoder SAM-H to ONNX format:

```bash
python -m samexporter.export_encoder --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.encoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.encoder.quant.onnx \
    --use-preprocess
```

- Convert decoder SAM-H to ONNX format:

```bash
python -m samexporter.export_decoder --checkpoint original_models/sam_vit_h_4b8939.pth \
    --output output_models/sam_vit_h_4b8939.decoder.onnx \
    --model-type vit_h \
    --quantize-out output_models/sam_vit_h_4b8939.decoder.quant.onnx \
    --return-single-mask
```

Remove `--return-single-mask` if you want to return multiple masks.

- Inference using the exported ONNX model:

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

```bash
python -m samexporter.inference \
    --encoder_model output_models/sam_vit_h_4b8939.encoder.onnx \
    --decoder_model output_models/sam_vit_h_4b8939.decoder.onnx \
    --image images/plants.png \
    --prompt images/plants_prompt2.json \
    --output output_images/plants_02.png \
    --show
```

![plants_02](https://raw.githubusercontent.com/vietanhdev/samexporter/main/sample_outputs/plants_02.png)


**Short options:**

- Convert all Meta's models to ONNX format:

```bash
bash convert_all_meta_sam.sh
```

- Convert MobileSAM to ONNX format:

```bash
bash convert_mobile_sam.sh
```

## Tips

- Use "quantized" models for faster inference and smaller model size. However, the accuracy may be lower than the original models.
- SAM-B is the most lightweight model, but it has the lowest accuracy. SAM-H is the most accurate model, but it has the largest model size. SAM-M is a good trade-off between accuracy and model size.

## AnyLabeling

This package was originally developed for auto labeling feature in [AnyLabeling](https://github.com/vietanhdev/anylabeling) project. However, you can use it for other purposes.

[![](https://user-images.githubusercontent.com/18329471/236625792-07f01838-3f69-48b0-a12e-30bad27bd921.gif)](https://youtu.be/5qVJiYNX5Kk)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
