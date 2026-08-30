import argparse
import json
import pathlib

import cv2
import numpy as np

from samexporter.efficient_sam_onnx import EfficientSAMONNX
from samexporter.sam2_onnx import SegmentAnything2ONNX
from samexporter.sam3_onnx import SegmentAnything3ONNX
from samexporter.sam_onnx import SegmentAnythingONNX


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a split SAM ONNX model")
    parser.add_argument(
        "--encoder_model",
        default="output_models/sam_vit_h_4b8939.encoder.onnx",
        help="Path to the ONNX encoder model",
    )
    parser.add_argument(
        "--decoder_model",
        default="output_models/sam_vit_h_4b8939.decoder.onnx",
        help="Path to the ONNX decoder model",
    )
    parser.add_argument(
        "--language_encoder_model",
        default=None,
        help="Path to the ONNX language encoder model (SAM3 only)",
    )
    parser.add_argument(
        "--text_prompt",
        default=None,
        help="SAM3 text prompt; overrides a text mark in the prompt JSON",
    )
    parser.add_argument("--image", default="images/truck.jpg")
    parser.add_argument("--prompt", default="images/truck_prompt.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--sam_variant",
        choices=("sam", "sam2", "sam3", "efficient_sam"),
        default="sam",
        help="ONNX model family",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="SAM3 detection confidence threshold",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.7,
        help="SAM3 mask- or box-IoU threshold for duplicate suppression",
    )
    parser.add_argument(
        "--sam3_nms_mode",
        choices=("mask", "box", "none"),
        default="mask",
        help="SAM3 duplicate suppression: official-style mask IoU, box IoU, or none",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=None,
        help="Maximum SAM3 instances to retain after ranking",
    )
    parser.add_argument(
        "--sam3_output_mode",
        choices=("auto", "all"),
        default="auto",
        help=(
            "SAM3 auto returns the best prompt-overlapping instance when geometry "
            "is present; all preserves open-vocabulary instance discovery"
        ),
    )
    parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Comma-separated ONNX Runtime provider names or aliases, for example "
            "tensorrt,cuda,cpu or openvino,cpu"
        ),
    )
    return parser


def load_model(args):
    if args.sam_variant == "sam":
        return SegmentAnythingONNX(
            args.encoder_model, args.decoder_model, providers=args.providers
        )
    if args.sam_variant == "sam2":
        return SegmentAnything2ONNX(
            args.encoder_model, args.decoder_model, providers=args.providers
        )
    if args.sam_variant == "efficient_sam":
        return EfficientSAMONNX(
            args.encoder_model, args.decoder_model, providers=args.providers
        )
    return SegmentAnything3ONNX(
        args.encoder_model,
        args.decoder_model,
        args.language_encoder_model,
        providers=args.providers,
    )


def get_text_prompt(prompt: list[dict], override: str | None) -> str:
    if override:
        return override
    for mark in prompt:
        if mark.get("type") == "text" and isinstance(mark.get("data"), str):
            return mark["data"]
    return "visual"


def visualize(image: np.ndarray, masks: np.ndarray, prompt, variant: str) -> np.ndarray:
    visualized = image.copy()
    if variant == "sam3":
        palette = (
            (255, 99, 71),
            (60, 179, 113),
            (0, 165, 255),
            (238, 130, 238),
            (255, 215, 0),
            (255, 144, 30),
            (147, 112, 219),
            (64, 224, 208),
        )
        for index, instance_mask in enumerate(masks[:, 0]):
            binary_mask = instance_mask.astype(bool)
            color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
            visualized[binary_mask] = (
                visualized[binary_mask].astype(np.float32) * 0.5 + color * 0.5
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(visualized, contours, -1, tuple(map(int, color)), 2)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                moments = cv2.moments(largest)
                if moments["m00"]:
                    center = (
                        int(moments["m10"] / moments["m00"]),
                        int(moments["m01"] / moments["m00"]),
                    )
                    cv2.putText(
                        visualized,
                        str(index + 1),
                        center,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
    else:
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        model_masks = masks[:, 0] if variant == "efficient_sam" else masks[0]
        for mask in model_masks:
            combined_mask |= mask > 0.0
        color = np.asarray([255, 0, 0], dtype=np.uint8)
        visualized[combined_mask] = (
            visualized[combined_mask].astype(np.float32) * 0.5 + color * 0.5
        ).astype(np.uint8)

    for mark in prompt:
        if mark.get("type") == "point":
            color = (0, 255, 0) if mark.get("label") == 1 else (0, 0, 255)
            cv2.circle(visualized, tuple(map(int, mark["data"])), 10, color, -1)
        elif mark.get("type") == "rectangle":
            x1, y1, x2, y2 = map(int, mark["data"])
            cv2.rectangle(visualized, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return visualized


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    with open(args.prompt, encoding="utf-8") as prompt_file:
        prompt = json.load(prompt_file)
    if not isinstance(prompt, list):
        raise ValueError("Prompt JSON must contain a list of marks")

    model = load_model(args)
    if args.sam_variant == "sam3":
        embedding = model.encode(
            image, text_prompt=get_text_prompt(prompt, args.text_prompt)
        )
        has_geometry = any(
            mark.get("type") in ("point", "rectangle") for mark in prompt
        )
        prefer_prompted_region = args.sam3_output_mode == "auto" and has_geometry
        max_instances = args.max_instances
        if prefer_prompted_region and max_instances is None:
            max_instances = 1
        masks = model.predict_masks(
            embedding,
            prompt,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            nms_mode=args.sam3_nms_mode,
            max_instances=max_instances,
            prefer_prompted_region=prefer_prompted_region,
        )
        print(f"SAM3 instances retained: {len(masks)}")
    else:
        embedding = model.encode(image)
        masks = model.predict_masks(embedding, prompt)

    visualized = visualize(image, masks, prompt, args.sam_variant)
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), visualized):
            raise OSError(f"Could not write output image: {output_path}")
    if args.show:
        cv2.imshow("Result", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
