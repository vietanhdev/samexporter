import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from samexporter.inference import visualize


def test_visualize_only_blends_masked_pixels():
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    masks = np.zeros((1, 1, 20, 20), dtype=np.float32)
    masks[0, 0, 5:10, 5:10] = 1

    result = visualize(image, masks, [], "sam")

    np.testing.assert_array_equal(result[0, 0], image[0, 0])
    assert not np.array_equal(result[6, 6], image[6, 6])


def test_visualize_sam3_instances_have_distinct_colors():
    image = np.full((40, 40, 3), 100, dtype=np.uint8)
    masks = np.zeros((2, 1, 40, 40), dtype=np.bool_)
    masks[0, 0, 2:12, 2:12] = True
    masks[1, 0, 25:35, 25:35] = True

    result = visualize(image, masks, [], "sam3")

    np.testing.assert_array_equal(result[20, 20], image[20, 20])
    assert not np.array_equal(result[5, 5], result[30, 30])
