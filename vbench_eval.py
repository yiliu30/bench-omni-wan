#!/usr/bin/env python3
"""Run VBench evaluation with a fixed RNG seed."""

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import packaging
import packaging.version
import torch


DEFAULT_VBENCH_EVALUATE = Path("/home/yiliu7/workspace/VBench/evaluate.py")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def patch_numpy_sctypes() -> None:
    if hasattr(np, "sctypes"):
        return

    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.object_, np.str_, np.bytes_],
    }


def patch_pkg_resources_packaging() -> None:
    try:
        import pkg_resources
    except ImportError:
        return

    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper around VBench evaluate.py with fixed seeding."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("EVAL_SEED", "42")),
        help="Seed used for Python, NumPy, and PyTorch RNGs.",
    )
    parser.add_argument(
        "--vbench-evaluate",
        default=os.environ.get("VBENCH_EVALUATE"),
        help="Path to VBench evaluate.py (default: $VBENCH_EVALUATE or legacy path).",
    )
    parser.add_argument(
        "evaluate_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to VBench evaluate.py. Prefix with --.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_args = args.evaluate_args
    if evaluate_args and evaluate_args[0] == "--":
        evaluate_args = evaluate_args[1:]

    set_seed(args.seed)
    patch_numpy_sctypes()
    patch_pkg_resources_packaging()

    evaluate_path = Path(args.vbench_evaluate) if args.vbench_evaluate else DEFAULT_VBENCH_EVALUATE
    if not evaluate_path.is_file():
        raise SystemExit(
            f"VBench evaluate.py not found: {evaluate_path}. "
            "Set VBENCH_EVALUATE or pass --vbench-evaluate."
        )
    sys.argv = [str(evaluate_path), *evaluate_args]
    runpy.run_path(str(evaluate_path), run_name="__main__")


if __name__ == "__main__":
    main()
