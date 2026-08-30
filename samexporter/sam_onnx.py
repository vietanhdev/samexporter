from copy import deepcopy

import cv2
import numpy as np
import onnxruntime
from PIL import Image

from samexporter.prompts import geometric_prompt_arrays
from samexporter.runtime import get_onnx_providers


class SegmentAnythingONNX:
    """Segmentation model using Segment Anything (SAM)"""

    def __init__(self, encoder_model_path, decoder_model_path, providers=None) -> None:
        self.target_size = 1024

        # Load models
        providers = get_onnx_providers(providers)
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=providers
        )
        encoder_input = self.encoder_session.get_inputs()[0]
        self.encoder_input_name = encoder_input.name
        self.encoder_input_shape = encoder_input.shape
        self.encoder_input_rank = len(encoder_input.shape)
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=providers
        )
        self.decoder_output_names = [
            model_output.name for model_output in self.decoder_session.get_outputs()
        ]

    def get_input_points(self, prompt):
        """Get input points"""
        return geometric_prompt_arrays(prompt)

    def run_encoder(self, encoder_inputs):
        """Run encoder"""
        output = self.encoder_session.run(None, encoder_inputs)
        image_embedding = output[0]
        return image_embedding

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def apply_coords(self, coords: np.ndarray, original_size, target_length):
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def run_decoder(self, image_embedding, original_size, resized_size, prompt):
        """Run decoder"""
        input_points, input_labels = self.get_input_points(prompt)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[
            None, :, :
        ]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, original_size, self.target_size
        ).astype(np.float32)

        # Create an empty mask input and an indicator for no mask.
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(original_size, dtype=np.float32),
        }
        outputs = self.decoder_session.run(None, decoder_inputs)

        # The segment-anything 1.0 ONNX wrapper converts a tensor-derived crop
        # size to Python ``int`` during tracing. That freezes the export to the
        # dummy image's landscape ratio. Prefer low-resolution logits and do
        # the standard resize/crop/resize here so every aspect ratio remains
        # correct (including portrait and extreme panoramas).
        if "low_res_masks" in self.decoder_output_names:
            masks = outputs[self.decoder_output_names.index("low_res_masks")]
            return self.postprocess_masks(masks, original_size, resized_size)

        masks = outputs[0]
        if masks.shape[-2:] != tuple(original_size):
            return self.resize_masks(masks, original_size)
        return masks

    def postprocess_masks(self, masks, original_size, resized_size):
        """Apply SAM's aspect-ratio-aware mask postprocessing outside ONNX."""
        output_masks = []
        for batch in masks:
            batch_masks = []
            for mask in batch:
                upscaled = cv2.resize(
                    mask,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                cropped = upscaled[: resized_size[0], : resized_size[1]]
                batch_masks.append(
                    cv2.resize(
                        cropped,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
            output_masks.append(batch_masks)
        return np.asarray(output_masks)

    @staticmethod
    def resize_masks(masks, original_size):
        return np.asarray(
            [
                [
                    cv2.resize(
                        mask,
                        (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for mask in batch
                ]
                for batch in masks
            ]
        )

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform the masks back to the original image size."""
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        """
        Calculate embedding and metadata for a single image.
        """
        if cv_image is None or cv_image.ndim != 3 or cv_image.shape[2] != 3:
            raise ValueError("Expected a non-empty BGR image with three channels")
        original_size = cv_image.shape[:2]
        resized_size = self.get_preprocess_shape(*original_size, self.target_size)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # Match official ResizeLongestSide.apply_image exactly. OpenCV's
        # sampler can move thin/ambiguous point-prompt boundaries enough to
        # select a different mask, especially after large downscales.
        rgb_image = np.asarray(
            Image.fromarray(rgb_image).resize(
                (resized_size[1], resized_size[0]),
                resample=Image.Resampling.BILINEAR,
            )
        )

        if self.encoder_input_rank == 3:
            # Encoder exported with --use-preprocess: dynamic HWC RGB input.
            input_image = rgb_image.astype(np.float32)
        elif self.encoder_input_rank == 4:
            # Encoder exported without preprocessing: normalized, padded NCHW.
            input_image = rgb_image.astype(np.float32)
            mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)
            input_image = (input_image - mean) / std
            input_image = input_image.transpose(2, 0, 1)
            pad_h = self.target_size - resized_size[0]
            pad_w = self.target_size - resized_size[1]
            input_image = np.pad(input_image, ((0, 0), (0, pad_h), (0, pad_w)))
            input_image = input_image[None].astype(np.float32)
        else:
            raise ValueError(
                f"Unsupported SAM encoder input rank: {self.encoder_input_rank}"
            )

        encoder_inputs = {
            self.encoder_input_name: input_image,
        }
        image_embedding = self.run_encoder(encoder_inputs)
        return {
            "image_embedding": image_embedding,
            "original_size": original_size,
            "resized_size": resized_size,
        }

    def predict_masks(self, embedding, prompt):
        """
        Predict masks for a single image.
        """
        masks = self.run_decoder(
            embedding["image_embedding"],
            embedding["original_size"],
            embedding["resized_size"],
            prompt,
        )

        return masks
