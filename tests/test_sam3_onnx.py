"""Tests for samexporter.sam3_onnx – SAM3 ONNX inference wrapper.

All tests mock onnxruntime.InferenceSession so no actual ONNX files are
needed.  The actual model shapes are confirmed by inspecting the exported
ONNX models with onnxruntime and are documented inline.

Actual model I/O (from sam3_decoder.onnx inspection):
  Image encoder  input : 'image'            [3, 1008, 1008]   uint8
  Image encoder  outputs: 6 × float tensors (vision_pos_enc_{0,1,2}, backbone_fpn_{0,1,2})

  Language encoder input : 'tokens'          [1, 32]           int64
  Language encoder outputs:
    [0] 'text_attention_mask'  [1, 32]        bool
    [1] 'text_memory'          [32, 1, 256]   float
    [2] 'text_embeds'          [32, 1, 1024]  float

  Decoder inputs (after onnxsim – some were simplified away):
    original_height   scalar  int64
    original_width    scalar  int64
    vision_pos_enc_2  [1, 256, 72, 72]     float   (0 and 1 were removed)
    backbone_fpn_0    [1, 256, 288, 288]   float
    backbone_fpn_1    [1, 256, 144, 144]   float
    backbone_fpn_2    [1, 256, 72, 72]     float
    language_mask     [1, 32]              bool
    language_features [32, 1, 256]         float
    box_coords        [1, 1, 4]            float
    box_labels        [1, 1]               int64
    box_masks         [1, 1]               bool
  Decoder outputs:
    [0] boxes   (N, 4)           float
    [1] scores  (N,)             float
    [2] masks   (N, 1, H, W)     bool
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure the samexporter package is importable from the tests directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samexporter.sam3_onnx import (
    SegmentAnything3ONNX,
    SAM3ImageEncoder,
    SAM3ImageDecoder,
    SAM3LanguageEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_session(input_specs, run_return=None):
    """Return a MagicMock that mimics an onnxruntime.InferenceSession.

    Parameters
    ----------
    input_specs:
        List of ``(name, shape, type_str)`` tuples describing model inputs.
    run_return:
        Value returned by ``session.run()``.  Defaults to ``[np.zeros(1)]``.
    """
    sess = MagicMock()
    input_mocks = []
    for name, shape, type_str in input_specs:
        m = MagicMock()
        m.name = name
        m.shape = shape
        m.type = type_str
        input_mocks.append(m)
    sess.get_inputs.return_value = input_mocks
    sess.run.return_value = run_return if run_return is not None else [np.zeros(1)]
    return sess


# Decoder input specs matching the actual simplified ONNX model.
_DECODER_INPUT_SPECS = [
    ("original_height",   [],             "tensor(int64)"),
    ("original_width",    [],             "tensor(int64)"),
    ("vision_pos_enc_2",  [1, 256, 72, 72], "tensor(float)"),
    ("backbone_fpn_0",    [1, 256, 288, 288], "tensor(float)"),
    ("backbone_fpn_1",    [1, 256, 144, 144], "tensor(float)"),
    ("backbone_fpn_2",    [1, 256, 72, 72],   "tensor(float)"),
    ("language_mask",     [1, 32],            "tensor(bool)"),
    ("language_features", [32, 1, 256],       "tensor(float)"),
    ("box_coords",        [1, 1, 4],          "tensor(float)"),
    ("box_labels",        [1, 1],             "tensor(int64)"),
    ("box_masks",         [1, 1],             "tensor(bool)"),
]

_ENCODER_INPUT_SPECS = [
    ("image", [3, 1008, 1008], "tensor(uint8)"),
]

_LANG_INPUT_SPECS = [
    ("tokens", [1, 32], "tensor(int64)"),
]


# ---------------------------------------------------------------------------
# SegmentAnything3ONNX integration tests (mocked sessions)
# ---------------------------------------------------------------------------

class TestSAM3OnnxRectanglePrompt(unittest.TestCase):
    """Verify that a rectangle prompt is correctly converted to box_coords."""

    @patch("onnxruntime.InferenceSession")
    def test_rectangle_box_coords(self, MockSession):
        # Give the encoder a trivially small mock; decoder gets realistic specs.
        enc_sess = _make_mock_session(
            _ENCODER_INPUT_SPECS,
            run_return=[np.zeros((1, 256, 288, 288))] * 3
                      + [np.zeros((1, 256, 288, 288))] * 3,
        )
        H, W = 100, 200
        dec_sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            # Decoder returns (boxes, scores, masks) – i.e. [0]=boxes, [1]=scores, [2]=masks.
            run_return=[
                np.zeros((1, 4), dtype=np.float32),    # boxes
                np.array([0.9], dtype=np.float32),      # scores
                np.ones((1, 1, H, W), dtype=np.bool_),  # masks
            ],
        )
        MockSession.side_effect = [enc_sess, dec_sess]

        model = SegmentAnything3ONNX("enc.onnx", "dec.onnx")

        # Build a minimal embedding (bypass encode()).
        embedding = {
            "original_size": (H, W),
            "vision_pos_enc_0": np.zeros((1, 256, 288, 288), dtype=np.float32),
            "vision_pos_enc_1": np.zeros((1, 256, 144, 144), dtype=np.float32),
            "vision_pos_enc_2": np.zeros((1, 256, 72, 72),  dtype=np.float32),
            "backbone_fpn_0":   np.zeros((1, 256, 288, 288), dtype=np.float32),
            "backbone_fpn_1":   np.zeros((1, 256, 144, 144), dtype=np.float32),
            "backbone_fpn_2":   np.zeros((1, 256, 72, 72),  dtype=np.float32),
            "language_mask":     np.zeros((1, 32), dtype=np.bool_),
            "language_features": np.zeros((32, 1, 256), dtype=np.float32),
            "language_embeds":   np.zeros((32, 1, 1024), dtype=np.float32),
        }

        # Rectangle: [x1=50, y1=20, x2=150, y2=80] in a 100×200 image.
        # cx = (50+150)/2 / 200 = 100/200 = 0.5
        # cy = (20+80)/2  / 100 = 50/100  = 0.5
        # w  = (150-50)   / 200 = 0.5
        # h  = (80-20)    / 100 = 0.6
        prompt = [{"type": "rectangle", "data": [50, 20, 150, 80]}]
        model.predict_masks(embedding, prompt)

        args, _kwargs = dec_sess.run.call_args
        inputs = args[1]
        self.assertIn("box_coords", inputs)
        np.testing.assert_allclose(
            inputs["box_coords"],
            np.array([[[0.5, 0.5, 0.5, 0.6]]], dtype=np.float32),
            atol=1e-5,
        )
        # box_masks should be False (real box provided).
        self.assertFalse(inputs["box_masks"].all())

    @patch("onnxruntime.InferenceSession")
    def test_empty_prompt_uses_dummy_box(self, MockSession):
        """With no marks the decoder should receive a dummy box (box_masks=True)."""
        enc_sess = _make_mock_session(_ENCODER_INPUT_SPECS)
        H, W = 64, 64
        dec_sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            run_return=[
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 1, H, W), dtype=np.bool_),
            ],
        )
        MockSession.side_effect = [enc_sess, dec_sess]

        model = SegmentAnything3ONNX("enc.onnx", "dec.onnx")
        embedding = {
            "original_size": (H, W),
            "vision_pos_enc_0": np.zeros((1, 256, 18, 18), dtype=np.float32),
            "vision_pos_enc_1": np.zeros((1, 256, 9, 9),   dtype=np.float32),
            "vision_pos_enc_2": np.zeros((1, 256, 5, 5),   dtype=np.float32),
            "backbone_fpn_0":   np.zeros((1, 256, 18, 18), dtype=np.float32),
            "backbone_fpn_1":   np.zeros((1, 256, 9, 9),   dtype=np.float32),
            "backbone_fpn_2":   np.zeros((1, 256, 5, 5),   dtype=np.float32),
            "language_mask":     np.zeros((1, 32), dtype=np.bool_),
            "language_features": np.zeros((32, 1, 256), dtype=np.float32),
            "language_embeds":   np.zeros((32, 1, 1024), dtype=np.float32),
        }
        model.predict_masks(embedding, [])

        args, _kwargs = dec_sess.run.call_args
        inputs = args[1]
        # When no prompt is given, box_masks should be True (dummy box).
        self.assertTrue(inputs["box_masks"].all())


class TestSAM3OnnxPointPrompt(unittest.TestCase):
    """Verify that a point prompt is converted to a small box."""

    @patch("onnxruntime.InferenceSession")
    def test_point_box_coords(self, MockSession):
        H, W = 200, 400
        enc_sess = _make_mock_session(_ENCODER_INPUT_SPECS)
        dec_sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            run_return=[
                np.zeros((1, 4), dtype=np.float32),
                np.array([0.8], dtype=np.float32),
                np.ones((1, 1, H, W), dtype=np.bool_),
            ],
        )
        MockSession.side_effect = [enc_sess, dec_sess]

        model = SegmentAnything3ONNX("enc.onnx", "dec.onnx")
        embedding = {
            "original_size": (H, W),
            **{k: np.zeros(1) for k in (
                "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2",
                "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
                "language_embeds",
            )},
            "language_mask":     np.zeros((1, 32), dtype=np.bool_),
            "language_features": np.zeros((32, 1, 256), dtype=np.float32),
        }
        # Point at pixel (200, 100): cx=200/400=0.5, cy=100/200=0.5
        prompt = [{"type": "point", "data": [200, 100], "label": 1}]
        model.predict_masks(embedding, prompt)

        args, _kwargs = dec_sess.run.call_args
        inputs = args[1]
        np.testing.assert_allclose(
            inputs["box_coords"],
            np.array([[[0.5, 0.5, 0.01, 0.01]]], dtype=np.float32),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# SAM3ImageEncoder tests
# ---------------------------------------------------------------------------

class TestSAM3ImageEncoder(unittest.TestCase):

    @patch("onnxruntime.InferenceSession")
    def test_prepare_input_uint8_model(self, MockSession):
        """uint8 model → prepare_input returns (C, H, W) uint8."""
        MockSession.return_value = _make_mock_session(_ENCODER_INPUT_SPECS)
        enc = SAM3ImageEncoder("enc.onnx")

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = enc.prepare_input(image)

        self.assertEqual(tensor.shape, (3, 1008, 1008))
        self.assertEqual(tensor.dtype, np.uint8)

    @patch("onnxruntime.InferenceSession")
    def test_prepare_input_float_model(self, MockSession):
        """float model → prepare_input returns normalised float32 in [-1, 1]."""
        MockSession.return_value = _make_mock_session(
            [("image", [1, 3, 1008, 1008], "tensor(float)")]
        )
        enc = SAM3ImageEncoder("enc.onnx")

        # All-255 image → normalised value = (1.0 - 0.5) / 0.5 = 1.0
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        tensor = enc.prepare_input(image)

        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor.max()), 1.0, places=5)
        self.assertAlmostEqual(float(tensor.min()), 1.0, places=5)

    @patch("onnxruntime.InferenceSession")
    def test_input_dimensions_3d(self, MockSession):
        """3-D model shape [3, H, W] is parsed correctly."""
        MockSession.return_value = _make_mock_session(
            [("image", [3, 512, 512], "tensor(uint8)")]
        )
        enc = SAM3ImageEncoder("enc.onnx")
        self.assertEqual(enc.input_height, 512)
        self.assertEqual(enc.input_width, 512)

    @patch("onnxruntime.InferenceSession")
    def test_input_dimensions_4d(self, MockSession):
        """4-D model shape [1, 3, H, W] (legacy) is parsed correctly."""
        MockSession.return_value = _make_mock_session(
            [("image", [1, 3, 1008, 1008], "tensor(uint8)")]
        )
        enc = SAM3ImageEncoder("enc.onnx")
        self.assertEqual(enc.input_height, 1008)
        self.assertEqual(enc.input_width, 1008)


# ---------------------------------------------------------------------------
# SAM3ImageDecoder tests
# ---------------------------------------------------------------------------

class TestSAM3ImageDecoder(unittest.TestCase):

    @patch("onnxruntime.InferenceSession")
    def test_returns_masks_scores_boxes(self, MockSession):
        """Decoder __call__ returns (masks, scores, boxes) in that order."""
        H, W = 100, 100
        boxes  = np.zeros((2, 4),     dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        masks  = np.ones((2, 1, H, W), dtype=np.bool_)

        sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            run_return=[boxes, scores, masks],  # ONNX order: boxes, scores, masks
        )
        MockSession.return_value = sess

        dec = SAM3ImageDecoder("dec.onnx")
        original_size = (H, W)
        dummy = np.zeros(1, dtype=np.float32)
        dummy_bool = np.zeros((1, 32), dtype=np.bool_)
        dummy_feat = np.zeros((32, 1, 256), dtype=np.float32)

        ret_masks, ret_scores, ret_boxes = dec(
            original_size,
            dummy, dummy, dummy,  # vision_pos_enc 0,1,2
            dummy, dummy, dummy,  # backbone_fpn 0,1,2
            dummy_bool,           # language_mask
            dummy_feat,           # language_features
            None,                 # language_embeds (not needed by decoder)
            np.zeros((1, 1, 4), dtype=np.float32),
            np.ones((1, 1), dtype=np.int64),
            np.zeros((1, 1), dtype=np.bool_),
        )

        # Verify order: (masks, scores, boxes)
        np.testing.assert_array_equal(ret_masks, masks)
        np.testing.assert_array_equal(ret_scores, scores)
        np.testing.assert_array_equal(ret_boxes, boxes)

    @patch("onnxruntime.InferenceSession")
    def test_dummy_language_inputs_when_none(self, MockSession):
        """Decoder supplies correct dummy tensors when language inputs are None."""
        sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            run_return=[
                np.zeros((1, 4), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.zeros((1, 1, 10, 10), dtype=np.bool_),
            ],
        )
        MockSession.return_value = sess

        dec = SAM3ImageDecoder("dec.onnx")
        dec(
            (10, 10),
            None, None, np.zeros((1, 256, 2, 2), dtype=np.float32),
            np.zeros((1, 256, 4, 4), dtype=np.float32),
            np.zeros((1, 256, 2, 2), dtype=np.float32),
            np.zeros((1, 256, 2, 2), dtype=np.float32),
            None,  # language_mask  → should become zeros (1, 32) bool
            None,  # language_features → should become zeros (32, 1, 256) float
            None,  # language_embeds  (not in decoder inputs, filtered out)
            np.zeros((1, 1, 4), dtype=np.float32),
            np.ones((1, 1), dtype=np.int64),
            np.ones((1, 1), dtype=np.bool_),
        )

        args, _kwargs = sess.run.call_args
        inputs = args[1]
        # Correct dummy shapes.
        self.assertEqual(inputs["language_mask"].shape, (1, 32))
        self.assertEqual(inputs["language_mask"].dtype, np.bool_)
        self.assertEqual(inputs["language_features"].shape, (32, 1, 256))
        self.assertEqual(inputs["language_features"].dtype, np.float32)
        # vision_pos_enc_0/1 not in simplified model – must not be forwarded.
        self.assertNotIn("vision_pos_enc_0", inputs)
        self.assertNotIn("vision_pos_enc_1", inputs)

    @patch("onnxruntime.InferenceSession")
    def test_bool_masks_returned_unmodified(self, MockSession):
        """Bool masks from the decoder are returned without dtype conversion."""
        H, W = 8, 8
        masks = np.array([[[[True, False] * 4] * H]], dtype=np.bool_)
        sess = _make_mock_session(
            _DECODER_INPUT_SPECS,
            run_return=[
                np.zeros((1, 4), dtype=np.float32),
                np.array([0.95], dtype=np.float32),
                masks,
            ],
        )
        MockSession.return_value = sess

        dec = SAM3ImageDecoder("dec.onnx")
        ret_masks, _, _ = dec(
            (H, W),
            None, None, np.zeros((1, 256, 2, 2), dtype=np.float32),
            np.zeros((1, 256, 4, 4), dtype=np.float32),
            np.zeros((1, 256, 2, 2), dtype=np.float32),
            np.zeros((1, 256, 2, 2), dtype=np.float32),
            np.zeros((1, 32), dtype=np.bool_),
            np.zeros((32, 1, 256), dtype=np.float32),
            None,
            np.zeros((1, 1, 4), dtype=np.float32),
            np.ones((1, 1), dtype=np.int64),
            np.ones((1, 1), dtype=np.bool_),
        )
        self.assertEqual(ret_masks.dtype, np.bool_)
        np.testing.assert_array_equal(ret_masks, masks)


# ---------------------------------------------------------------------------
# SegmentAnything3ONNX.encode – no language encoder
# ---------------------------------------------------------------------------

class TestSAM3OnnxEncodeNoLanguageEncoder(unittest.TestCase):

    @patch("onnxruntime.InferenceSession")
    def test_encode_without_language_encoder_sets_none(self, MockSession):
        """encode() sets language keys to None when no language encoder is given."""
        enc_sess = _make_mock_session(
            _ENCODER_INPUT_SPECS,
            run_return=[np.zeros(1)] * 6,
        )
        dec_sess = _make_mock_session(_DECODER_INPUT_SPECS)
        MockSession.side_effect = [enc_sess, dec_sess]

        model = SegmentAnything3ONNX("enc.onnx", "dec.onnx")
        # Minimal BGR image
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        embedding = model.encode(image)

        self.assertIn("language_mask", embedding)
        self.assertIn("language_features", embedding)
        self.assertIn("language_embeds", embedding)
        self.assertIsNone(embedding["language_mask"])
        self.assertIsNone(embedding["language_features"])
        self.assertIsNone(embedding["language_embeds"])


if __name__ == "__main__":
    unittest.main()
