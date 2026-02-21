import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samexporter.onnx_utils import ImageEncoderOnnxModel


class TestOnnxUtils(unittest.TestCase):
    def test_preprocess_logic(self):
        # Mock SAM model and image encoder
        mock_sam = MagicMock()
        mock_sam.image_encoder.img_size = 1024

        model = ImageEncoderOnnxModel(mock_sam, use_preprocess=True)

        # Input image (H, W, C)
        image = torch.zeros((100, 200, 3))

        processed = model.preprocess(image)

        # Expected shape: (1, 3, 1024, 1024)
        self.assertEqual(processed.shape, (1, 3, 1024, 1024))
        self.assertEqual(processed.dtype, torch.float)


if __name__ == "__main__":
    unittest.main()
