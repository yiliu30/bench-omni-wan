#!/usr/bin/env python3
"""Build side-by-side frames from five matched I2V videos per recipe."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PAIRED = ROOT / "results/i2v_lightning_50_upstream_vae"
NO_FALLBACK = ROOT / "results/i2v_lightning_50_no_fallback"
OUTPUT = ROOT / "results/i2v_three_way_frames"
PAIRS = (0, 10, 20, 30, 49)
FRAMES = (20, 60)
RECIPES = (
    (PAIRED / "bf16_flash", "BF16 Flash"),
    (PAIRED / "mxfp4_sage_flash4", "MXFP4 Sage + fallback"),
    (NO_FALLBACK / "mxfp4_sage_no_fallback", "MXFP4 Sage no fallback"),
)


def selected_frames(path: Path) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    found = {}
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count in FRAMES:
                if frame.shape != (640, 640, 3):
                    raise ValueError(f"Unexpected frame dimensions in {path}: {frame.shape}")
                found[count] = frame
            count += 1
    finally:
        capture.release()
    if count != 81 or set(found) != set(FRAMES):
        raise ValueError(f"Expected 81 frames including {FRAMES} in {path}; found {count}")
    return found


def main() -> None:
    paired = json.loads((PAIRED / "manifest.json").read_text(encoding="utf-8"))
    other = json.loads((NO_FALLBACK / "manifest.json").read_text(encoding="utf-8"))
    for index in PAIRS:
        first = paired["items"][index]
        second = other["items"][index]
        if any(first[key] != second[key] for key in ("index", "stem", "prompt", "sha256")):
            raise ValueError(f"Mismatched input pair {index}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in PAIRS:
        stem = paired["items"][index]["stem"]
        videos = [selected_frames(directory / f"{stem}.mp4") for directory, _ in RECIPES]
        for frame_index in FRAMES:
            row = np.full((680, 1920, 3), 245, dtype=np.uint8)
            for column, ((_, label), frames) in enumerate(zip(RECIPES, videos)):
                x = column * 640
                cv2.putText(row, f"{label} | pair {stem} frame {frame_index}",
                            (x + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
                row[40:, x:x + 640] = frames[frame_index]
            path = OUTPUT / f"pair_{stem}_frame_{frame_index:03}.png"
            if not cv2.imwrite(str(path), row):
                raise RuntimeError(f"Failed to save {path}")
            rows.append(row)

    overview = OUTPUT / "overview.png"
    if not cv2.imwrite(str(overview), np.concatenate(rows, axis=0)):
        raise RuntimeError(f"Failed to save {overview}")
    print(f"Saved {len(rows)} three-way rows and overview to {OUTPUT}")


if __name__ == "__main__":
    main()
