import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samexporter.sam_onnx import SegmentAnythingONNX
from samexporter.sam2_onnx import SegmentAnything2ONNX

class TestSAMVariants(unittest.TestCase):
    
    @patch('onnxruntime.InferenceSession')
    def test_sam1_logic(self, mock_session):
        # Setup mock session
        mock_sess_instance = MagicMock()
        # Mock input details for SAM1
        mock_input = MagicMock()
        mock_input.name = "image"
        mock_input.shape = [1, 3, 1024, 1024]
        mock_sess_instance.get_inputs.return_value = [mock_input]
        mock_session.return_value = mock_sess_instance
        
        model = SegmentAnythingONNX("dummy_enc.onnx", "dummy_dec.onnx")
        
        # Test encode
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_sess_instance.run.return_value = [np.zeros((1, 256, 64, 64))]
        embedding = model.encode(image)
        self.assertIn("image_embedding", embedding)
        
        # Test predict_masks
        prompt = [{"type": "point", "data": [50, 50], "label": 1}]
        mock_sess_instance.run.return_value = [np.zeros((1, 1, 100, 100)), np.zeros(1), np.zeros(1)]
        masks = model.predict_masks(embedding, prompt)
        self.assertEqual(len(masks.shape), 4)

    @patch('onnxruntime.InferenceSession')
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
        mock_sess_instance.run.return_value = [np.zeros((1, 3, 256, 256)), np.array([[0.9, 0.1, 0.1]])]
        masks = model.predict_masks(embedding, prompt)
        self.assertEqual(len(masks.shape), 4)

if __name__ == '__main__':
    unittest.main()
