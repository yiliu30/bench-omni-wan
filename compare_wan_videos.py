#!/usr/bin/env python3
"""Compare Wan MP4 outputs against a BF16 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE = ROOT / "wan_bf16_t2v_out.mp4"
DEFAULT_CANDIDATES = {
    "ocp_mxfp4_direct": ROOT / "wan_t2v_a14b_ocp_mxfp4_direct_after_review_fix_1280x720_81f_40steps.mp4",
    "uos_only": ROOT / "wan_t2v_a14b_uos_only_after_review_fix_1280x720_81f_40steps.mp4",
    "pnq_only": ROOT / "wan_t2v_a14b_pnq_only_1280x720_81f_40steps.mp4",
    "hadamard_only": ROOT / "wan_t2v_a14b_hadamard_only_1280x720_81f_40steps.mp4",
    "uos_pnq": ROOT / "wan_t2v_a14b_uos_pnq_1280x720_81f_40steps.mp4",
    "uos_hadamard": ROOT / "wan_t2v_a14b_uos_hadamard_1280x720_81f_40steps.mp4",
    "mxattention_full": ROOT / "wan_t2v_a14b_mxattention_full_after_review_fix_1280x720_81f_40steps.mp4",
    "mxfp4_hw": ROOT / "wan_t2v_a14b_mxfp4_hw_attn_1280x720_81f_40steps.mp4",
}
DEFAULT_OUTPUT_PREFIX = ROOT / "wan_mxattention_diff_report"


@dataclass(frozen=True)
class Video:
    path: Path
    frames: np.ndarray
    fps: float | None
    size: tuple[int, int] | None
    duration: float | None


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be MODE=PATH")
    mode, path = value.split("=", 1)
    mode = mode.strip()
    if not mode:
        raise argparse.ArgumentTypeError("candidate mode must be non-empty")
    return mode, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        default=None,
        help="Candidate video as MODE=PATH. May be repeated.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--lpips-net", default="alex", choices=("alex", "vgg", "squeeze"))
    parser.add_argument(
        "--lpips-device",
        default="auto",
        help="LPIPS device: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Compute MAE/MSE/PSNR/SSIM only.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Also compare the baseline to itself and include it in the report.",
    )
    return parser.parse_args()


def load_video(path: Path) -> Video:
    if not path.exists():
        raise FileNotFoundError(path)
    meta = iio.immeta(path)
    frames = np.stack([np.asarray(frame) for frame in iio.imiter(path)], axis=0)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"{path} decoded to unexpected shape {frames.shape}")
    frames = frames[..., :3]
    return Video(
        path=path,
        frames=frames,
        fps=float(meta["fps"]) if "fps" in meta else None,
        size=tuple(meta["size"]) if "size" in meta else None,
        duration=float(meta["duration"]) if "duration" in meta else None,
    )


def validate_compatible(reference: Video, candidate: Video, mode: str) -> None:
    if reference.frames.shape != candidate.frames.shape:
        raise ValueError(
            f"{mode}: frame tensors differ: "
            f"baseline={reference.frames.shape}, candidate={candidate.frames.shape}"
        )
    if reference.size != candidate.size:
        raise ValueError(f"{mode}: size differs: baseline={reference.size}, candidate={candidate.size}")
    if reference.fps is not None and candidate.fps is not None and abs(reference.fps - candidate.fps) > 1e-6:
        raise ValueError(f"{mode}: fps differs: baseline={reference.fps}, candidate={candidate.fps}")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if np.isposinf(array).all():
        return {
            "mean": math.inf,
            "min": math.inf,
            "max": math.inf,
            "p50": math.inf,
            "p95": math.inf,
        }
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def psnr_from_mse(mse: float) -> float:
    if mse == 0.0:
        return math.inf
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def _predefine_torchvision_nms() -> None:
    try:
        import torch

        library = torch.library.Library("torchvision", "DEF")
        library.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def build_lpips_model(net: str, device_arg: str) -> tuple[Any, str]:
    import torch
    try:
        import lpips
    except RuntimeError as exc:
        if "torchvision::nms" not in str(exc):
            raise
        _predefine_torchvision_nms()
        import lpips

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    model = lpips.LPIPS(net=net).eval().to(device)
    return model, device


def lpips_distance(model: Any, device: str, reference: np.ndarray, candidate: np.ndarray) -> float:
    import torch

    def to_tensor(frame: np.ndarray) -> torch.Tensor:
        array = frame.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device)

    with torch.inference_mode():
        return float(model(to_tensor(reference), to_tensor(candidate)).item())


def compare_videos(
    mode: str,
    reference: Video,
    candidate: Video,
    lpips_model: Any | None,
    lpips_device: str | None,
) -> dict[str, Any]:
    validate_compatible(reference, candidate, mode)

    ref_float = reference.frames.astype(np.float32) / 255.0
    cand_float = candidate.frames.astype(np.float32) / 255.0
    abs_diff = np.abs(ref_float - cand_float)
    sq_diff = np.square(ref_float - cand_float)

    frame_metrics: list[dict[str, float | int]] = []
    mae_values: list[float] = []
    mse_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for index in range(reference.frames.shape[0]):
        mae = float(abs_diff[index].mean())
        mse = float(sq_diff[index].mean())
        psnr = psnr_from_mse(mse)
        ssim = float(
            structural_similarity(
                ref_float[index],
                cand_float[index],
                channel_axis=2,
                data_range=1.0,
            )
        )
        metrics: dict[str, float | int] = {
            "frame": index,
            "mae": mae,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim,
        }
        if lpips_model is not None and lpips_device is not None:
            lpips_value = lpips_distance(lpips_model, lpips_device, reference.frames[index], candidate.frames[index])
            metrics["lpips"] = lpips_value
            lpips_values.append(lpips_value)

        frame_metrics.append(metrics)
        mae_values.append(mae)
        mse_values.append(mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    aggregate: dict[str, Any] = {
        "mode": mode,
        "candidate": str(candidate.path),
        "frames": int(reference.frames.shape[0]),
        "width": int(reference.frames.shape[2]),
        "height": int(reference.frames.shape[1]),
        "fps": reference.fps,
        "duration": reference.duration,
        "mae": summarize(mae_values),
        "mse": summarize(mse_values),
        "psnr_db": summarize(psnr_values),
        "ssim": summarize(ssim_values),
    }
    if lpips_values:
        aggregate["lpips"] = summarize(lpips_values)

    return {
        "aggregate": aggregate,
        "per_frame": frame_metrics,
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "candidate",
        "frames",
        "width",
        "height",
        "fps",
        "duration",
        "mae_mean",
        "mse_mean",
        "psnr_db_mean",
        "ssim_mean",
        "lpips_mean",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            aggregate = result["aggregate"]
            writer.writerow(
                {
                    "mode": aggregate["mode"],
                    "candidate": aggregate["candidate"],
                    "frames": aggregate["frames"],
                    "width": aggregate["width"],
                    "height": aggregate["height"],
                    "fps": aggregate["fps"],
                    "duration": aggregate["duration"],
                    "mae_mean": aggregate["mae"]["mean"],
                    "mse_mean": aggregate["mse"]["mean"],
                    "psnr_db_mean": aggregate["psnr_db"]["mean"],
                    "ssim_mean": aggregate["ssim"]["mean"],
                    "lpips_mean": aggregate.get("lpips", {}).get("mean"),
                }
            )


def main() -> None:
    args = parse_args()
    candidates = dict(args.candidate) if args.candidate else dict(DEFAULT_CANDIDATES)
    if args.self_check:
        candidates = {"bf16_self_check": args.baseline, **candidates}

    baseline = load_video(args.baseline)
    lpips_model = None
    lpips_device = None
    if not args.skip_lpips:
        lpips_model, lpips_device = build_lpips_model(args.lpips_net, args.lpips_device)

    results = []
    for mode, path in candidates.items():
        print(f"Comparing {mode}: {path}")
        candidate = load_video(path)
        result = compare_videos(mode, baseline, candidate, lpips_model, lpips_device)
        results.append(result)
        aggregate = result["aggregate"]
        lpips_mean = aggregate.get("lpips", {}).get("mean")
        print(
            f"  PSNR={aggregate['psnr_db']['mean']:.4f} dB "
            f"SSIM={aggregate['ssim']['mean']:.6f} "
            f"LPIPS={lpips_mean if lpips_mean is not None else 'skipped'}"
        )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "baseline": str(baseline.path),
                "lpips_net": None if args.skip_lpips else args.lpips_net,
                "lpips_device": lpips_device,
                "results": results,
            },
            indent=2,
        )
    )
    write_csv(csv_path, results)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
