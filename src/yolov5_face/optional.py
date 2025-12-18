from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


class OptionalDependencyError(ImportError):
    """Raised when an optional extra is required but not installed."""


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    """Import `module` or raise an actionable error recommending an extra."""

    root = module.split(".", 1)[0]
    if importlib.util.find_spec(root) is None:
        raise OptionalDependencyError(
            f"Missing optional dependency '{root}' required for {purpose}. "
            f"Install via `pip install yolov5-face[{extra}]`."
        )

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or root
        raise OptionalDependencyError(
            f"Missing optional dependency '{missing}' required for {purpose}. "
            f"Install via `pip install yolov5-face[{extra}]`."
        ) from exc
    except Exception as exc:
        raise OptionalDependencyError(
            f"Dependency '{module}' was found but failed to import (broken binary/env). "
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc
