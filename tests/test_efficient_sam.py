from unittest.mock import MagicMock, patch

import numpy as np

from samexporter.efficient_sam_onnx import EfficientSAMONNX


def model_input(name, shape):
    value = MagicMock()
    value.name = name
    value.shape = shape
    return value


class TestEfficientSAMONNX:
    @patch("onnxruntime.InferenceSession")
    def test_rgb_input_and_best_mask_selection(self, inference_session):
        encoder = MagicMock()
        encoder.get_inputs.return_value = [
            model_input("batched_images", ["batch", 3, "height", "width"])
        ]
        encoder.run.return_value = [np.zeros((1, 256, 64, 64), np.float32)]

        decoder = MagicMock()
        decoder.get_inputs.return_value = [
            model_input("image_embeddings", [1, 256, 64, 64]),
            model_input("batched_point_coords", [1, 1, "points", 2]),
            model_input("batched_point_labels", [1, 1, "points"]),
            model_input("orig_im_size", [2]),
        ]
        masks = np.zeros((1, 1, 3, 2, 3), np.float32)
        masks[0, 0, 2] = 1
        decoder.run.return_value = [
            masks,
            np.array([[[0.1, 0.2, 0.9]]], np.float32),
        ]
        inference_session.side_effect = [encoder, decoder]

        model = EfficientSAMONNX("encoder.onnx", "decoder.onnx")
        image = np.zeros((2, 3, 3), np.uint8)
        image[..., 0] = 255  # OpenCV blue becomes RGB red.
        embedding = model.encode(image)
        encoded_input = encoder.run.call_args.args[1]["batched_images"]
        assert encoded_input.shape == (1, 3, 2, 3)
        assert encoded_input[0, 0, 0, 0] == 0
        assert encoded_input[0, 2, 0, 0] == 1

        result = model.predict_masks(
            embedding, [{"type": "rectangle", "data": [0, 0, 2, 1]}]
        )
        assert result.shape == (1, 1, 2, 3)
        assert result.all()
        decoder_inputs = decoder.run.call_args.args[1]
        np.testing.assert_array_equal(
            decoder_inputs["batched_point_labels"], [[[2.0, 3.0]]]
        )

    @patch("onnxruntime.InferenceSession")
    def test_rejects_unknown_decoder_contract(self, inference_session):
        encoder = MagicMock()
        encoder.get_inputs.return_value = [
            model_input("batched_images", [1, 3, -1, -1])
        ]
        decoder = MagicMock()
        decoder.get_inputs.return_value = [model_input("unknown", [1])]
        inference_session.side_effect = [encoder, decoder]
        model = EfficientSAMONNX("encoder.onnx", "decoder.onnx")

        try:
            model.predict_masks(
                {"image_embedding": np.zeros(1), "original_size": (10, 10)},
                [{"type": "point", "data": [1, 1], "label": 1}],
            )
        except ValueError as error:
            assert "Unsupported EfficientSAM decoder inputs" in str(error)
        else:
            raise AssertionError("Expected incompatible decoder to be rejected")
