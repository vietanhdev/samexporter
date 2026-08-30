import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samexporter.sam2_onnx import SegmentAnything2ONNX
from samexporter.sam_onnx import SegmentAnythingONNX


class TestSAMVariants(unittest.TestCase):
    @patch("onnxruntime.InferenceSession")
    def test_sam1_logic(self, mock_session):
        # Setup mock session
        mock_sess_instance = MagicMock()
        # Mock input details for SAM1
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_input.shape = [1, 3, 1024, 1024]
        mock_sess_instance.get_inputs.return_value = [mock_input]
        mock_output = MagicMock()
        mock_output.name = "low_res_masks"
        mock_sess_instance.get_outputs.return_value = [
            MagicMock(name="masks"),
            MagicMock(name="iou_predictions"),
            mock_output,
        ]
        mock_session.return_value = mock_sess_instance

        model = SegmentAnythingONNX("dummy_enc.onnx", "dummy_dec.onnx")

        # Test encode
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_sess_instance.run.return_value = [np.zeros((1, 256, 64, 64))]
        embedding = model.encode(image)
        self.assertIn("image_embedding", embedding)

        # Test predict_masks
        prompt = [{"type": "point", "data": [50, 50], "label": 1}]
        mock_sess_instance.run.return_value = [
            np.zeros((1, 1, 100, 100)),
            np.zeros(1),
            np.zeros((1, 1, 256, 256)),
        ]
        masks = model.predict_masks(embedding, prompt)
        self.assertEqual(len(masks.shape), 4)

    @patch("onnxruntime.InferenceSession")
    def test_sam1_embedded_preprocess_preserves_portrait_aspect_and_rgb(
        self, mock_session
    ):
        encoder = MagicMock()
        encoder_input = MagicMock()
        encoder_input.name = "input_image"
        encoder_input.shape = ["image_height", "image_width", 3]
        encoder.get_inputs.return_value = [encoder_input]
        encoder.run.return_value = [np.zeros((1, 256, 64, 64), np.float32)]
        decoder = MagicMock()
        mock_session.side_effect = [encoder, decoder]

        model = SegmentAnythingONNX("encoder.onnx", "decoder.onnx")
        image = np.zeros((1200, 600, 3), np.uint8)
        image[..., 0] = 255
        embedding = model.encode(image)

        input_image = encoder.run.call_args.args[1]["input_image"]
        self.assertEqual(input_image.shape, (1024, 512, 3))
        self.assertEqual(input_image[0, 0].tolist(), [0.0, 0.0, 255.0])
        self.assertEqual(embedding["resized_size"], (1024, 512))

    @patch("onnxruntime.InferenceSession")
    def test_sam1_postprocess_uses_portrait_crop(self, mock_session):
        encoder = MagicMock()
        encoder_input = MagicMock()
        encoder_input.name = "input_image"
        encoder_input.shape = ["image_height", "image_width", 3]
        encoder.get_inputs.return_value = [encoder_input]

        decoder = MagicMock()
        decoder.get_outputs.return_value = [
            MagicMock(name="masks"),
            MagicMock(name="iou_predictions"),
            MagicMock(name="low_res_masks"),
        ]
        mock_session.side_effect = [encoder, decoder]
        model = SegmentAnythingONNX("encoder.onnx", "decoder.onnx")

        low_res = np.zeros((1, 1, 256, 256), np.float32)
        low_res[:, :, :, :128] = 1
        result = model.postprocess_masks(
            low_res, original_size=(1200, 600), resized_size=(1024, 512)
        )
        self.assertEqual(result.shape, (1, 1, 1200, 600))
        self.assertGreater(float(result.mean()), 0.99)

    @patch("onnxruntime.InferenceSession")
    def test_sam2_logic(self, mock_session):
        # Setup mock session
        mock_sess_instance = MagicMock()
        # Mock input details for SAM2 encoder
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_input.shape = [1, 3, 1024, 1024]
        mock_sess_instance.get_inputs.return_value = [mock_input]
        # SAM2 encoder outputs 3 features
        mock_sess_instance.run.return_value = [np.zeros(1), np.zeros(1), np.zeros(1)]
        mock_session.return_value = mock_sess_instance

        model = SegmentAnything2ONNX("dummy_enc.onnx", "dummy_dec.onnx")

        # Test encode
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        embedding = model.encode(image)
        self.assertIn("high_res_feats_0", embedding)
        self.assertIn("image_embedding", embedding)

        # Test predict_masks
        prompt = [{"type": "rectangle", "data": [10, 10, 50, 50]}]
        # SAM2 decoder outputs masks and scores
        mock_sess_instance.run.return_value = [
            np.zeros((1, 3, 256, 256)),
            np.array([[0.9, 0.1, 0.1]]),
        ]
        masks = model.predict_masks(embedding, prompt)
        self.assertEqual(len(masks.shape), 4)


if __name__ == "__main__":
    unittest.main()
