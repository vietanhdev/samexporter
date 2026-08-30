from typing import Any

import cv2
import numpy as np
import onnxruntime

from samexporter.clip_tokenizer import tokenize
from samexporter.runtime import get_onnx_providers


class SegmentAnything3ONNX:
    """Segmentation model using Segment Anything 3 (SAM3)"""

    def __init__(
        self,
        image_encoder_path,
        decoder_model_path,
        language_encoder_path=None,
        providers=None,
    ) -> None:
        providers = get_onnx_providers(providers)
        self.image_encoder = SAM3ImageEncoder(image_encoder_path, providers)
        self.language_encoder = None
        if language_encoder_path:
            self.language_encoder = SAM3LanguageEncoder(
                language_encoder_path, providers
            )
        self.decoder = SAM3ImageDecoder(decoder_model_path, providers)

    def encode(self, cv_image: np.ndarray, text_prompt=None) -> dict[str, Any]:
        """Encode an image (and optional text prompt) into an embedding dict.

        Parameters
        ----------
        cv_image:
            BGR uint8 image as returned by ``cv2.imread``.
        text_prompt:
            Natural-language description of the target object.
            Falls back to ``"visual"`` when *None* (model-default).

        Returns
        -------
        dict with keys:
            original_size, vision_pos_enc_{0,1,2}, backbone_fpn_{0,1,2},
            language_mask, language_features, language_embeds.
        """
        if cv_image is None or cv_image.ndim != 3 or cv_image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        original_size = cv_image.shape[:2]
        image_encoder_outputs = self.image_encoder(cv_image)

        embedding: dict[str, Any] = {
            "vision_pos_enc_0": image_encoder_outputs[0],
            "vision_pos_enc_1": image_encoder_outputs[1],
            "vision_pos_enc_2": image_encoder_outputs[2],
            "backbone_fpn_0": image_encoder_outputs[3],
            "backbone_fpn_1": image_encoder_outputs[4],
            "backbone_fpn_2": image_encoder_outputs[5],
            "original_size": original_size,
            # Pre-fill language keys as None; overwritten below when a
            # language encoder is available.
            "language_mask": None,
            "language_features": None,
            "language_embeds": None,
        }

        text_prompt = text_prompt or "visual"
        if self.language_encoder is not None:
            lang_outputs = self.language_encoder(text_prompt)
            # lang_outputs indices:
            #   [0] text_attention_mask  – bool  [1, seq_len]
            #   [1] text_memory          – float [seq_len, 1, 256]
            #   [2] text_embeds          – float [seq_len, 1, 1024]
            embedding["language_mask"] = lang_outputs[0]
            embedding["language_features"] = lang_outputs[1]
            embedding["language_embeds"] = lang_outputs[2]

        return embedding

    def predict_masks(
        self,
        embedding: dict[str, Any],
        prompt,
        confidence_threshold: float = 0.5,
        nms_threshold: float | None = 0.7,
        nms_mode: str = "mask",
        max_instances: int | None = None,
        prefer_prompted_region: bool = False,
    ) -> np.ndarray:
        """Run the decoder for the given geometric prompt.

        Parameters
        ----------
        embedding:
            Dict returned by :meth:`encode`.
        prompt:
            List of mark dicts, each with keys ``"type"`` (``"rectangle"``
            or ``"point"``) and ``"data"``.
        confidence_threshold:
            Minimum score to keep a detection.  Detections with score below
            this value are discarded.  Defaults to ``0.5``.
        nms_threshold:
            IoU threshold used to suppress duplicate queries. Set to ``None``
            to retain threshold-only behavior.
        nms_mode:
            ``"mask"`` follows the official SAM3 mask-IoU NMS behavior;
            ``"box"`` is faster for very large masks; ``"none"`` disables NMS.
        max_instances:
            Optional maximum number of masks, after ranking and NMS.
        prefer_prompted_region:
            Rank masks overlapping positive points/rectangles ahead of remote
            concept matches. Useful when geometry is intended as a selection.

        Returns
        -------
        Boolean mask array of shape ``(N, 1, H, W)`` where *N* is the number
        of detected objects and *H* × *W* is the original image resolution.
        """
        original_size = embedding["original_size"]
        box_coords = []
        box_labels = []

        for index, mark in enumerate(prompt):
            if not isinstance(mark, dict):
                raise ValueError(f"Prompt mark {index} must be an object")
            mark_type = mark.get("type")
            if mark_type == "text":
                continue
            data = mark.get("data")
            if mark_type == "rectangle":
                if not isinstance(data, (list, tuple)) or len(data) != 4:
                    raise ValueError(
                        f"Rectangle mark {index} must contain [x1, y1, x2, y2]"
                    )
                x1, y1, x2, y2 = map(float, data)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Rectangle mark {index} must have positive area")
                cx = (x1 + x2) / 2.0 / original_size[1]
                cy = (y1 + y2) / 2.0 / original_size[0]
                w = (x2 - x1) / original_size[1]
                h = (y2 - y1) / original_size[0]
                box_coords.append([cx, cy, w, h])
                box_labels.append(1)
            elif mark_type == "point":
                if not isinstance(data, (list, tuple)) or len(data) != 2:
                    raise ValueError(f"Point mark {index} must contain [x, y]")
                label = mark.get("label")
                if label not in (0, 1):
                    raise ValueError(f"Point mark {index} label must be 0 or 1")
                x, y = map(float, data)
                cx = x / original_size[1]
                cy = y / original_size[0]
                # Point is represented as a very small box (1 % of image).
                box_coords.append([cx, cy, 0.01, 0.01])
                box_labels.append(label)
            else:
                raise ValueError(
                    f"Unsupported prompt type at mark {index}: {mark_type!r}"
                )

        mark_count = len(box_coords)
        capacity = self.decoder.geometric_prompt_capacity
        if capacity is not None and mark_count > capacity:
            raise ValueError(
                f"SAM3 decoder accepts at most {capacity} geometric prompts; "
                "re-export it with a larger --max-geometric-prompts value"
            )
        tensor_length = capacity or max(mark_count, 1)
        # SAM3's Prompt contract is sequence-first: [num_marks, batch, C].
        # ONNX traces its internal geometry attention at a fixed token count,
        # so exported decoders use padded slots and mark them True in box_masks.
        box_coords_np = np.zeros((tensor_length, 1, 4), dtype=np.float32)
        box_labels_np = np.ones((tensor_length, 1), dtype=np.int64)
        box_masks_np = np.ones((1, tensor_length), dtype=np.bool_)
        if mark_count:
            box_coords_np[:mark_count, 0] = np.asarray(box_coords, dtype=np.float32)
            box_labels_np[:mark_count, 0] = np.asarray(box_labels, dtype=np.int64)
            box_masks_np[0, :mark_count] = False

        masks, scores, boxes = self.decoder(
            original_size,
            embedding["vision_pos_enc_0"],
            embedding["vision_pos_enc_1"],
            embedding["vision_pos_enc_2"],
            embedding["backbone_fpn_0"],
            embedding["backbone_fpn_1"],
            embedding["backbone_fpn_2"],
            embedding["language_mask"],
            embedding["language_features"],
            embedding["language_embeds"],
            box_coords_np,
            box_labels_np,
            box_masks_np,
        )

        masks, _scores, _boxes = self.select_detections(
            masks,
            scores,
            boxes,
            prompt,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            nms_mode=nms_mode,
            max_instances=max_instances,
            prefer_prompted_region=prefer_prompted_region,
        )
        return masks

    @staticmethod
    def select_detections(
        masks: np.ndarray,
        scores: np.ndarray,
        boxes: np.ndarray,
        prompt,
        *,
        confidence_threshold: float,
        nms_threshold: float | None,
        nms_mode: str,
        max_instances: int | None,
        prefer_prompted_region: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter and rank raw SAM3 detections without model dependencies."""
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if nms_threshold is not None and not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1 or None")
        if nms_mode not in ("mask", "box", "none"):
            raise ValueError("nms_mode must be 'mask', 'box', or 'none'")
        if max_instances is not None and max_instances < 1:
            raise ValueError("max_instances must be at least 1 or None")

        masks = np.asarray(masks)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if not (len(masks) == len(scores) == len(boxes)):
            raise ValueError("SAM3 masks, scores, and boxes must have equal lengths")

        keep = np.flatnonzero(np.isfinite(scores) & (scores > confidence_threshold))
        if not len(keep):
            return (
                np.zeros((0,) + masks.shape[1:], dtype=masks.dtype),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )
        masks, scores, boxes = masks[keep], scores[keep], boxes[keep]

        order = np.argsort(-scores, kind="stable")
        masks, scores, boxes = masks[order], scores[order], boxes[order]
        if nms_mode != "none" and nms_threshold is not None and len(boxes) > 1:
            if nms_mode == "mask":
                keep = SegmentAnything3ONNX._mask_nms(masks, scores, nms_threshold)
            else:
                keep = SegmentAnything3ONNX._box_nms(boxes, scores, nms_threshold)
            masks, scores, boxes = masks[keep], scores[keep], boxes[keep]

        if prefer_prompted_region and len(masks) > 1:
            relevance = SegmentAnything3ONNX._prompt_relevance(masks, boxes, prompt)
            order = np.lexsort((-scores, -relevance))
            masks, scores, boxes = masks[order], scores[order], boxes[order]

        if max_instances is not None:
            masks = masks[:max_instances]
            scores = scores[:max_instances]
            boxes = boxes[:max_instances]
        return masks, scores, boxes

    @staticmethod
    def _box_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """Return score-ordered indices after standard XYXY box NMS."""
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []
        while len(order):
            current = int(order[0])
            kept.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            intersection_width = np.maximum(
                0.0,
                np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest]),
            )
            intersection_height = np.maximum(
                0.0,
                np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest]),
            )
            intersection = intersection_width * intersection_height
            union = areas[current] + areas[rest] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = rest[iou <= threshold]
        return np.asarray(kept, dtype=np.int64)

    @staticmethod
    def _mask_nms(
        masks: np.ndarray, scores: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return score-ordered indices after bit-packed mask-IoU NMS.

        SAM3's official NMS uses mask overlap. Masks are sampled to at most
        288 pixels on their longest side (the native SAM3 mask-head scale),
        then packed into bits so large original-resolution outputs do not make
        duplicate suppression memory-bound.
        """
        binary = np.asarray(masks, dtype=bool)
        if binary.ndim == 4 and binary.shape[1] == 1:
            binary = binary[:, 0]
        if binary.ndim != 3:
            raise ValueError("SAM3 masks must have shape (N, 1, H, W) or (N, H, W)")
        step = max(1, int(np.ceil(max(binary.shape[-2:]) / 288)))
        sampled = binary[:, ::step, ::step].reshape(len(binary), -1)
        packed = np.packbits(sampled, axis=1)
        bit_counts = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
            axis=1
        )
        areas = bit_counts[packed].sum(axis=1, dtype=np.int64)

        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []
        while len(order):
            current = int(order[0])
            kept.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            intersections = bit_counts[
                np.bitwise_and(packed[rest], packed[current])
            ].sum(axis=1, dtype=np.int64)
            unions = areas[current] + areas[rest] - intersections
            iou = np.divide(
                intersections,
                unions,
                out=np.zeros(len(rest), dtype=np.float64),
                where=unions > 0,
            )
            order = rest[iou <= threshold]
        return np.asarray(kept, dtype=np.int64)

    @staticmethod
    def _prompt_relevance(masks: np.ndarray, boxes: np.ndarray, prompt) -> np.ndarray:
        """Score positive-point/rectangle overlap for selection-style prompting."""
        relevance = np.zeros(len(masks), dtype=np.float32)
        mask_height, mask_width = masks.shape[-2:]
        for mark in prompt:
            if not isinstance(mark, dict) or mark.get("label", 1) != 1:
                continue
            data = mark.get("data")
            if (
                mark.get("type") == "point"
                and isinstance(data, (list, tuple))
                and len(data) == 2
            ):
                x = int(np.clip(round(float(data[0])), 0, mask_width - 1))
                y = int(np.clip(round(float(data[1])), 0, mask_height - 1))
                relevance += masks[:, 0, y, x].astype(np.float32) * 2.0
                relevance += (
                    (boxes[:, 0] <= x)
                    & (x <= boxes[:, 2])
                    & (boxes[:, 1] <= y)
                    & (y <= boxes[:, 3])
                ).astype(np.float32)
            elif (
                mark.get("type") == "rectangle"
                and isinstance(data, (list, tuple))
                and len(data) == 4
            ):
                prompt_box = np.asarray(data, dtype=np.float32)
                ix1 = np.maximum(boxes[:, 0], prompt_box[0])
                iy1 = np.maximum(boxes[:, 1], prompt_box[1])
                ix2 = np.minimum(boxes[:, 2], prompt_box[2])
                iy2 = np.minimum(boxes[:, 3], prompt_box[3])
                intersection = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                box_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
                    0.0, boxes[:, 3] - boxes[:, 1]
                )
                prompt_area = max(
                    0.0,
                    float(prompt_box[2] - prompt_box[0])
                    * float(prompt_box[3] - prompt_box[1]),
                )
                union = box_area + prompt_area - intersection
                relevance += np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                )
                center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
                center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
                relevance += (
                    (prompt_box[0] <= center_x)
                    & (center_x <= prompt_box[2])
                    & (prompt_box[1] <= center_y)
                    & (center_y <= prompt_box[3])
                ).astype(np.float32)
        return relevance

    def transform_masks(self, masks, original_size, transform_matrix):
        """No-op: SAM3 already outputs masks in original image resolution."""
        return masks


class SAM3ImageEncoder:
    """Runs the SAM3 image backbone ONNX model.

    Expected model input
    --------------------
    name  : ``"image"``
    shape : ``[3, 1008, 1008]``
    dtype : uint8 (the model includes normalization internally)
    """

    def __init__(self, path: str, providers=None) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=get_onnx_providers(providers)
        )
        encoder_input = self.session.get_inputs()[0]
        self.input_name: str = encoder_input.name
        self.input_shape = encoder_input.shape
        self.input_type: str = encoder_input.type
        # The model expects (C, H, W) without a batch dimension.
        # Shape is [3, H, W] so indices are 0=C, 1=H, 2=W.
        if len(self.input_shape) == 3:
            self.input_height: int = int(self.input_shape[1]) or 1008
            self.input_width: int = int(self.input_shape[2]) or 1008
        elif len(self.input_shape) >= 4:
            # Legacy: batched export [1, 3, H, W]
            self.input_height = int(self.input_shape[2]) or 1008
            self.input_width = int(self.input_shape[3]) or 1008
        else:
            self.input_height = 1008
            self.input_width = 1008

    def __call__(self, image: np.ndarray) -> list[np.ndarray]:
        input_tensor = self.prepare_input(image)
        return self.session.run(None, {self.input_name: input_tensor})

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR cv2 image to the encoder's expected tensor format."""
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(
            input_img,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # (H, W, C) → (C, H, W)
        input_img = input_img.transpose(2, 0, 1)

        if self.input_type == "tensor(float)":
            # Model does NOT include normalisation – apply it here.
            # Maps [0, 255] uint8 → [−1, 1] float32 via (x/255 − 0.5) / 0.5.
            input_tensor = ((input_img / 255.0) - 0.5) / 0.5
            input_tensor = input_tensor.astype(np.float32)
        else:
            # Model includes normalisation internally – pass raw uint8.
            input_tensor = input_img.astype(np.uint8)

        return input_tensor


class SAM3LanguageEncoder:
    """Runs the SAM3 language-encoder ONNX model.

    Expected model input
    --------------------
    name  : ``"tokens"``
    shape : ``[1, 32]``
    dtype : int64
    """

    def __init__(self, path: str, providers=None) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=get_onnx_providers(providers)
        )

    def __call__(self, text: str) -> list[np.ndarray]:
        tokens = tokenize([text], context_length=32)
        return self.session.run(None, {"tokens": tokens})


class SAM3ImageDecoder:
    """Runs the SAM3 decoder ONNX model.

    Expected output order (from ONNX export names):
        [0] boxes  – float (N, 4)
        [1] scores – float (N,)
        [2] masks  – bool  (N, 1, H, W)

    The ``__call__`` method returns ``(masks, scores, boxes)`` so that
    callers can unpack in a semantically natural order.
    """

    def __init__(self, path: str, providers=None) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=get_onnx_providers(providers)
        )
        self.input_names: list[str] = [i.name for i in self.session.get_inputs()]
        self.output_names: list[str] = [
            output.name for output in self.session.get_outputs()
        ]
        box_input = next(
            (item for item in self.session.get_inputs() if item.name == "box_coords"),
            None,
        )
        first_dimension = box_input.shape[0] if box_input is not None else None
        self.geometric_prompt_capacity = (
            first_dimension
            if isinstance(first_dimension, int) and first_dimension > 0
            else None
        )

    def __call__(
        self,
        original_size,
        vision_pos_enc_0,
        vision_pos_enc_1,
        vision_pos_enc_2,
        backbone_fpn_0,
        backbone_fpn_1,
        backbone_fpn_2,
        language_mask,
        language_features,
        language_embeds,
        box_coords,
        box_labels,
        box_masks,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs: dict[str, np.ndarray | None] = {
            "original_height": np.array(original_size[0], dtype=np.int64),
            "original_width": np.array(original_size[1], dtype=np.int64),
            "vision_pos_enc_0": vision_pos_enc_0,
            "vision_pos_enc_1": vision_pos_enc_1,
            "vision_pos_enc_2": vision_pos_enc_2,
            "backbone_fpn_0": backbone_fpn_0,
            "backbone_fpn_1": backbone_fpn_1,
            "backbone_fpn_2": backbone_fpn_2,
            "language_mask": language_mask,
            "language_features": language_features,
            "language_embeds": language_embeds,
            "box_coords": box_coords,
            "box_labels": box_labels,
            "box_masks": box_masks,
        }

        # Supply dummy language tensors when no language encoder was used.
        # Shapes match the actual ONNX model's expected inputs.
        if "language_mask" in self.input_names and inputs["language_mask"] is None:
            inputs["language_mask"] = np.zeros((1, 32), dtype=np.bool_)
        if (
            "language_features" in self.input_names
            and inputs["language_features"] is None
        ):
            # Shape: [seq_len, batch=1, feature_dim=256]
            inputs["language_features"] = np.zeros((32, 1, 256), dtype=np.float32)
        if "language_embeds" in self.input_names and inputs["language_embeds"] is None:
            # Shape: [seq_len, batch=1, embed_dim=1024]
            inputs["language_embeds"] = np.zeros((32, 1, 1024), dtype=np.float32)

        # Only forward inputs that the model actually expects (onnxsim may
        # have removed some during simplification, e.g. vision_pos_enc_0/1).
        model_inputs = {
            k: v for k, v in inputs.items() if k in self.input_names and v is not None
        }
        outputs = self.session.run(None, model_inputs)
        if "pred_logits" in self.output_names:
            raw = dict(zip(self.output_names, outputs))
            return self.postprocess_raw_outputs(raw, original_size)

        # Backward compatibility with older exports whose processor-based
        # graph returned already-filtered [boxes, scores, masks].
        return outputs[2], outputs[1], outputs[0]

    @staticmethod
    def postprocess_raw_outputs(raw, original_size):
        """Match the official Sam3Processor postprocessing outside ONNX."""
        logits = np.asarray(raw["pred_logits"])[0].squeeze(-1)
        presence = np.asarray(raw["presence_logit_dec"]).reshape(-1)[0]

        def sigmoid(value):
            return 1.0 / (1.0 + np.exp(-np.clip(value, -80, 80)))

        scores = sigmoid(logits) * sigmoid(presence)

        raw_masks = np.asarray(raw["pred_masks"])[0]
        height, width = original_size
        masks = np.stack(
            [
                sigmoid(
                    cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                )
                > 0.5
                for mask in raw_masks
            ]
        )[:, None]

        boxes_cxcywh = np.asarray(raw["pred_boxes"])[0]
        cx, cy, box_width, box_height = boxes_cxcywh.T
        boxes = np.stack(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ],
            axis=-1,
        )
        return masks, scores, boxes
