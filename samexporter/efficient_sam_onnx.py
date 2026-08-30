from typing import Any

import cv2
import numpy as np
import onnxruntime

from samexporter.prompts import geometric_prompt_arrays
from samexporter.runtime import get_onnx_providers


class EfficientSAMONNX:
    """EfficientSAM-S/Ti inference using official split ONNX models.

    The model contract follows the Apache-2.0 EfficientSAM reference exporter:
    a dynamic NCHW RGB image encoder and a prompt decoder accepting one or more
    point/box marks.
    """

    def __init__(
        self, encoder_model_path: str, decoder_model_path: str, providers=None
    ) -> None:
        providers = get_onnx_providers(providers)
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=providers
        )
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=providers
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_input_names = {
            model_input.name for model_input in self.decoder_session.get_inputs()
        }

    def encode(self, cv_image: np.ndarray) -> dict[str, Any]:
        if cv_image is None or cv_image.ndim != 3 or cv_image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        input_image = rgb_image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        image_embedding = self.encoder_session.run(
            None, {self.encoder_input_name: input_image}
        )[0]
        return {
            "image_embedding": image_embedding,
            "original_size": cv_image.shape[:2],
        }

    def predict_masks(self, embedding: dict[str, Any], prompt) -> np.ndarray:
        points, labels = geometric_prompt_arrays(prompt)
        decoder_inputs = {
            "image_embeddings": embedding["image_embedding"],
            "batched_point_coords": points[None, None],
            "batched_point_labels": labels[None, None],
            "orig_im_size": np.asarray(embedding["original_size"], dtype=np.int64),
        }
        missing = self.decoder_input_names - decoder_inputs.keys()
        if missing:
            raise ValueError(
                "Unsupported EfficientSAM decoder inputs: " + ", ".join(sorted(missing))
            )

        outputs = self.decoder_session.run(
            None,
            {name: decoder_inputs[name] for name in self.decoder_input_names},
        )
        masks, iou_predictions = outputs[:2]
        # Official shape: masks [B, queries, candidates, H, W], IoU
        # [B, queries, candidates]. Select the best candidate per query.
        if masks.ndim != 5 or iou_predictions.ndim != 3:
            raise ValueError(
                "Unexpected EfficientSAM decoder output shapes: "
                f"{masks.shape}, {iou_predictions.shape}"
            )
        best_indices = np.argmax(iou_predictions[0], axis=-1)
        best_masks = np.stack(
            [masks[0, query, best] for query, best in enumerate(best_indices)]
        )
        return best_masks[:, None]
