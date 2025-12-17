import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


NUMPY_GLOBALS: Tuple[object, ...] = (
    np.core.multiarray._reconstruct,
    np.core.multiarray.scalar,
)


def safe_global_context(globals_: Iterable[object]):
    ctx = getattr(torch.serialization, "safe_globals", None)
    if ctx is None:
        class _Fallback:
            def __init__(self, allowed):
                self.allowed = tuple(allowed)

            def __enter__(self):
                torch.serialization.add_safe_globals(self.allowed)

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

        return _Fallback(globals_)
    return ctx(globals_)


def ensure_yolo_modules():
    try:
        import yolov5_face as yvf  # noqa: F401
        import yolov5_face.models as yvf_models
    except Exception as err:
        raise SystemExit(f"Failed to import yolov5_face: {err}") from err
    sys.modules.setdefault("models", yvf_models)


def convert_checkpoint(src: Path, dst: Path) -> dict:
    if not src.exists():
        raise SystemExit(f"Input weights not found: {src}")

    with safe_global_context(NUMPY_GLOBALS):
        ensure_yolo_modules()
        checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    model = checkpoint.get("model")
    if model is None:
        raise SystemExit(f"Checkpoint {src} does not contain a 'model' entry.")

    model = model.float().eval().cpu()
    converted = {
        "state_dict": model.state_dict(),
        "model_cfg": getattr(model, "yaml", None),
        "names": getattr(model, "names", None),
        "stride": getattr(model, "stride", None),
    }

    torch.save(converted, dst)
    print(f"[convert] Saved modern checkpoint to {dst}")
    return converted


def verify_checkpoint(path: Path):
    from yolov5_face.models.yolo import Model

    checkpoint = torch.load(path, map_location="cpu")
    missing_keys = {"state_dict", "model_cfg"} - checkpoint.keys()
    if missing_keys:
        raise SystemExit(f"[verify] Missing keys {missing_keys} in {path}")

    cfg = checkpoint["model_cfg"]
    model = Model(cfg, ch=cfg.get("ch", 3), nc=cfg.get("nc"))
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        print(f"[verify] Warning: missing {missing}, unexpected {unexpected}")
    model.eval()
    print(f"[verify] Successfully instantiated Model from {path}")


def resolve_output_path(src: Path, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    stem = src.stem
    suffix = src.suffix or ".pt"
    return src.with_name(f"{stem}_modern{suffix}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy YOLOv5-face checkpoint into a modern torch.save-compatible file "
            "and verify it can be reloaded."
        )
    )
    parser.add_argument("weights", help="Path to the original YOLOv5-face .pt file.")
    parser.add_argument(
        "--output",
        help="Optional output filename. Defaults to <weights>_modern.pt in the same directory.",
    )
    args = parser.parse_args()

    src = Path(args.weights).expanduser()
    dst = resolve_output_path(src, args.output)

    converted = convert_checkpoint(src, dst)
    try:
        verify_checkpoint(dst)
    except Exception as err:
        raise SystemExit(f"[verify] Failed to reload {dst}: {err}") from err

    print(f"[done] Modern weights ready at {dst}")
    print(f"[done] Metadata: names={converted['names']}, stride={converted['stride']}")


if __name__ == "__main__":
    main()
