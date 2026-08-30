import sys
from pathlib import Path

_UPSTREAM_PATHS = {
    "sam1": "third_party/segment-anything",
    "sam2": "third_party/sam2",
    "sam3": "sam3",
}


def prefer_pinned_upstream(model_family: str) -> Path | None:
    """Prefer a checked-out pinned upstream while keeping wheel fallback.

    Source-tree users get the exact submodule revision recorded by this
    repository. Installed wheels do not contain the submodules and continue to
    resolve their separately installed upstream package.
    """
    relative_path = _UPSTREAM_PATHS.get(model_family)
    if relative_path is None:
        raise ValueError(f"Unknown upstream model family: {model_family}")
    repository_root = Path(__file__).resolve().parent.parent
    upstream_path = repository_root / relative_path
    if upstream_path.is_dir() and str(upstream_path) not in sys.path:
        sys.path.insert(0, str(upstream_path))
        return upstream_path
    return upstream_path if upstream_path.is_dir() else None
