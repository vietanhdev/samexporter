import pytest

from samexporter.prompts import geometric_prompt_arrays


def test_text_marks_are_ignored_when_geometric_prompt_exists():
    points, labels = geometric_prompt_arrays(
        [
            {"type": "text", "data": "leaf"},
            {"type": "point", "data": [4, 5], "label": 1},
        ]
    )
    assert points.tolist() == [[4.0, 5.0]]
    assert labels.tolist() == [1.0]


@pytest.mark.parametrize(
    "prompt, message",
    [
        ([], "At least one"),
        ([{"type": "point", "data": [1, 2], "label": 4}], "label"),
        ([{"type": "rectangle", "data": [2, 2, 1, 3]}], "positive area"),
        ([{"type": "polygon", "data": []}], "Unsupported"),
    ],
)
def test_invalid_prompts_fail_clearly(prompt, message):
    with pytest.raises(ValueError, match=message):
        geometric_prompt_arrays(prompt)
