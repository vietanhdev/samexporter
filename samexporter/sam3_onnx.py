from typing import Any

import cv2
import numpy as np
import onnxruntime


class SegmentAnything3ONNX:
    """Segmentation model using Segment Anything 3 (SAM3)"""

    def __init__(
        self,
        image_encoder_path,
        decoder_model_path,
        language_encoder_path=None,
    ) -> None:
        self.image_encoder = SAM3ImageEncoder(image_encoder_path)
        self.language_encoder = None
        if language_encoder_path:
            self.language_encoder = SAM3LanguageEncoder(language_encoder_path)
        self.decoder = SAM3ImageDecoder(decoder_model_path)

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

    def predict_masks(self, embedding: dict[str, Any], prompt) -> np.ndarray:
        """Run the decoder for the given geometric prompt.

        Parameters
        ----------
        embedding:
            Dict returned by :meth:`encode`.
        prompt:
            List of mark dicts, each with keys ``"type"`` (``"rectangle"``
            or ``"point"``) and ``"data"``.

        Returns
        -------
        Boolean mask array of shape ``(N, 1, H, W)`` where *N* is the number
        of detected objects and *H* × *W* is the original image resolution.
        """
        original_size = embedding["original_size"]
        box_coords = [0.0, 0.0, 0.0, 0.0]
        box_labels = [1]
        # box_masks: True  → dummy / no real box
        #            False → a real box is provided
        box_masks = [True]

        for mark in prompt:
            if mark["type"] == "rectangle":
                x1, y1, x2, y2 = mark["data"]
                cx = (x1 + x2) / 2.0 / original_size[1]
                cy = (y1 + y2) / 2.0 / original_size[0]
                w = (x2 - x1) / original_size[1]
                h = (y2 - y1) / original_size[0]
                box_coords = [cx, cy, w, h]
                box_masks = [False]
                break
            elif mark["type"] == "point":
                x, y = mark["data"]
                cx = x / original_size[1]
                cy = y / original_size[0]
                # Point is represented as a very small box (1 % of image).
                box_coords = [cx, cy, 0.01, 0.01]
                box_masks = [False]
                break

        box_coords_np = np.array(box_coords, dtype=np.float32).reshape(1, 1, 4)
        box_labels_np = np.array([box_labels], dtype=np.int64)
        box_masks_np = np.array([box_masks], dtype=np.bool_)

        masks, _scores, _boxes = self.decoder(
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

        return masks

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

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
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

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        try:
            from osam._models.yoloworld.clip import tokenize

            self._tokenize = tokenize
        except ImportError:
            self._tokenize = self._fallback_tokenize

    def _fallback_tokenize(self, texts, context_length: int = 32) -> np.ndarray:
        """Minimal CLIP-style tokeniser fallback (zeros = empty sequence).

        Warning: the model will produce near-random language features when
        this fallback is used.  Install ``osam`` for correct tokenisation.
        """
        return np.zeros((len(texts), context_length), dtype=np.int64)

    def __call__(self, text: str) -> list[np.ndarray]:
        tokens = self._tokenize([text], context_length=32)
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens, dtype=np.int64)
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

    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        self.input_names: list[str] = [i.name for i in self.session.get_inputs()]

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
        # ONNX export order: [0]=boxes, [1]=scores, [2]=masks
        # Return as (masks, scores, boxes) for caller convenience.
        return outputs[2], outputs[1], outputs[0]
