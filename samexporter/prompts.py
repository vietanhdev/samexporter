from collections.abc import Sequence

import numpy as np


def geometric_prompt_arrays(
    prompt: Sequence[dict], *, allow_empty: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Validate point/rectangle marks and return SAM coordinate/label arrays."""
    points: list[list[float]] = []
    labels: list[float] = []

    for index, mark in enumerate(prompt):
        if not isinstance(mark, dict):
            raise ValueError(f"Prompt mark {index} must be an object")
        mark_type = mark.get("type")
        if mark_type == "text":
            continue
        data = mark.get("data")
        if mark_type == "point":
            if not isinstance(data, (list, tuple)) or len(data) != 2:
                raise ValueError(f"Point mark {index} must contain [x, y]")
            label = mark.get("label")
            if label not in (0, 1):
                raise ValueError(f"Point mark {index} label must be 0 or 1")
            points.append([float(data[0]), float(data[1])])
            labels.append(float(label))
        elif mark_type == "rectangle":
            if not isinstance(data, (list, tuple)) or len(data) != 4:
                raise ValueError(
                    f"Rectangle mark {index} must contain [x1, y1, x2, y2]"
                )
            x1, y1, x2, y2 = (float(value) for value in data)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Rectangle mark {index} must have positive area")
            points.extend([[x1, y1], [x2, y2]])
            labels.extend([2.0, 3.0])
        else:
            raise ValueError(f"Unsupported prompt type at mark {index}: {mark_type!r}")

    if not points and not allow_empty:
        raise ValueError("At least one point or rectangle prompt is required")

    return (
        np.asarray(points, dtype=np.float32).reshape(-1, 2),
        np.asarray(labels, dtype=np.float32),
    )
