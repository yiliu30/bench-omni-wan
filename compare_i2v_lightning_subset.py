#!/usr/bin/env python3
"""Verify and compare paired VBench-I2V Lightning videos (not a VBench score)."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from i2v_lightning_subset import RECIPES, config, validate_video


def decoded(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode {path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != 81:
        raise ValueError(f"Expected 81 frames in {path}, found {len(frames)}")
    return frames


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    first = reference.astype(np.float32)
    second = candidate.astype(np.float32)
    mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    variance_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean_first * mean_first
    variance_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean_second * mean_second
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_first * mean_second
    numerator = (2 * mean_first * mean_second + 6.5025) * (2 * covariance + 58.5225)
    denominator = (mean_first**2 + mean_second**2 + 6.5025) * (
        variance_first + variance_second + 58.5225
    )
    return float(np.mean(numerator / denominator))


def compare(root: Path, items: list[dict] | None = None) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report_dir = root / "comparison"
    report_dir.mkdir(exist_ok=True)
    per_frame = []
    per_pair = []
    for item in manifest["items"] if items is None else items:
        videos = [root / recipe / f"{item['stem']}.mp4" for recipe in RECIPES]
        records = [root / recipe / f"{item['stem']}.json" for recipe in RECIPES]
        for record_path, recipe, video in zip(records, RECIPES, videos):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record.get("status") != "completed" or record.get("recipe") != recipe
                    or record.get("image_sha256") != item["sha256"]
                    or record.get("prompt") != item["prompt"]
                    or record.get("metadata_sha256") != manifest["metadata_sha256"]
                    or record.get("video") != validate_video(video)):
                raise ValueError(f"Invalid/mismatched pair: {record_path}")
        left, right = (decoded(video) for video in videos)
        errors = []
        similarities = []
        for frame_idx, (reference, candidate) in enumerate(zip(left, right)):
            mae = float(cv2.absdiff(reference, candidate).mean())
            similarity = ssim(reference, candidate)
            errors.append(mae)
            similarities.append(similarity)
            per_frame.append((item["index"], frame_idx, mae, similarity))
        for frame_idx in (0, 20, 40, 60, 80):
            sheet = np.full((680, 1280, 3), 245, dtype=np.uint8)
            cv2.putText(sheet, f"BF16 Flash | {item['stem']} frame {frame_idx}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            cv2.putText(sheet, f"MXFP4 Sage + fallback | {item['stem']} frame {frame_idx}", (652, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            sheet[40:, :640] = left[frame_idx]
            sheet[40:, 640:] = right[frame_idx]
            output = report_dir / f"pair_{item['stem']}_frame_{frame_idx:03}.png"
            if not cv2.imwrite(str(output), sheet):
                raise RuntimeError(f"Failed to write {output}")
        per_pair.append((item["index"], item["image_name"], item["image_type"],
                         float(np.mean(errors)), float(np.mean(similarities))))

    for filename, header, rows in (
        ("per_frame.csv", ("pair", "frame", "mae_0_255", "ssim"), per_frame),
        ("per_pair.csv", ("pair", "image_name", "image_type", "mean_mae_0_255", "mean_ssim"), per_pair),
    ):
        with (report_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    summary = {"pairs": len(per_pair), "frames_per_pair": 81,
               "mean_mae_0_255": float(np.mean([row[2] for row in per_frame])),
               "mean_ssim": float(np.mean([row[3] for row in per_frame])),
               "note": "Decoded-pixel similarity only; NOT VBench MUSIQ-SPAQ imaging_quality."}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", default="i2v_lightning_50.env")
    args = parser.parse_args()
    path = Path(args.env_file)
    cfg = config(path if path.is_absolute() else Path(__file__).parent / path)
    print(json.dumps(compare(Path(cfg["OUTPUT"])), indent=2))


if __name__ == "__main__":
    main()
