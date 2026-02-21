import argparse
import json
import pathlib
import sys

sys.path.append(".")

import cv2
import numpy as np

from samexporter.sam2_onnx import SegmentAnything2ONNX
from samexporter.sam3_onnx import SegmentAnything3ONNX
from samexporter.sam_onnx import SegmentAnythingONNX


def str2bool(v):
    return v.lower() in ("true", "1")


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--encoder_model",
    type=str,
    default="output_models/sam_vit_h_4b8939.encoder.onnx",
    help="Path to the ONNX encoder model",
)
argparser.add_argument(
    "--decoder_model",
    type=str,
    default="output_models/sam_vit_h_4b8939.decoder.onnx",
    help="Path to the ONNX decoder model",
)
argparser.add_argument(
    "--language_encoder_model",
    type=str,
    default=None,
    help="Path to the ONNX language encoder model (for SAM3)",
)
argparser.add_argument(
    "--text_prompt",
    type=str,
    default=None,
    help="Text prompt for SAM3 (e.g. 'truck'). Overrides any text entry in the prompt JSON.",
)
argparser.add_argument(
    "--image",
    type=str,
    default="images/truck.jpg",
    help="Path to the image",
)
argparser.add_argument(
    "--prompt",
    type=str,
    default="images/truck_prompt.json",
    help="Path to the image",
)
argparser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to the output image",
)
argparser.add_argument(
    "--show",
    action="store_true",
    help="Show the result",
)
argparser.add_argument(
    "--sam_variant",
    type=str,
    default="sam",
    help="Variant of SAM model. Options: sam, sam2, sam3",
)
args = argparser.parse_args()

model = None
if args.sam_variant == "sam":
    model = SegmentAnythingONNX(
        args.encoder_model,
        args.decoder_model,
    )
elif args.sam_variant == "sam2":
    model = SegmentAnything2ONNX(
        args.encoder_model,
        args.decoder_model,
    )
elif args.sam_variant == "sam3":
    model = SegmentAnything3ONNX(
        args.encoder_model,
        args.decoder_model,
        args.language_encoder_model,
    )

image = cv2.imread(args.image)
prompt = json.load(open(args.prompt))

text_prompt = None
if args.sam_variant == "sam3":
    # --text_prompt takes priority; fall back to any text entry in the JSON.
    if args.text_prompt:
        text_prompt = args.text_prompt
    else:
        for p in prompt:
            if p["type"] == "text":
                text_prompt = p["data"]
                break
    if text_prompt is None:
        text_prompt = "visual"

embedding = (
    model.encode(image, text_prompt=text_prompt)
    if args.sam_variant == "sam3"
    else model.encode(image)
)

masks = model.predict_masks(embedding, prompt)

# Merge masks
mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
if args.sam_variant == "sam3":
    # SAM3 returns bool (N, 1, H, W) – render all N detected instances.
    for i in range(masks.shape[0]):
        m = masks[i, 0]  # (H, W) bool
        mask[m] = [255, 0, 0]
else:
    # SAM1/SAM2 return raw logits (1, 3, H, W) – threshold at 0 (= sigmoid 0.5).
    for m in masks[0, :, :, :]:
        mask[m > 0.0] = [255, 0, 0]

# Binding image and mask
visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

# Draw the prompt points and rectangles.
for p in prompt:
    if p["type"] == "point":
        color = (
            (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
        )  # green for positive, red for negative
        cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
    elif p["type"] == "rectangle":
        cv2.rectangle(
            visualized,
            (p["data"][0], p["data"][1]),
            (p["data"][2], p["data"][3]),
            (0, 255, 0),
            2,
        )

if args.output is not None:
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, visualized)

if args.show:
    cv2.imshow("Result", visualized)
    cv2.waitKey(0)
