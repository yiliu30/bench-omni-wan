#!/usr/bin/env python3
"""Validate MXAttention-full common-10 generation artifacts and write provenance."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    return parser.parse_args()


def probe_video(path: Path) -> dict[str, int | float]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_frames"]),
        "fps": float(numerator) / float(denominator),
    }


def main() -> None:
    args = parse_args()
    prompts = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()][:10]
    if len(prompts) != 10:
        raise SystemExit("Expected at least 10 non-empty prompts")
    if not args.server_log.is_file():
        raise SystemExit(f"Missing server log: {args.server_log}")

    log = args.server_log.read_text(errors="replace")
    required = ("VLLM_MXATTENTION", "mxattention_full")
    missing_log_evidence = [item for item in required if item.lower() not in log.lower()]
    fallback_markers = ("falling back to sdpa", "hardware disabled")
    fallbacks = [marker for marker in fallback_markers if marker in log.lower()]
    if missing_log_evidence or fallbacks:
        details = []
        if missing_log_evidence:
            details.append("missing log evidence: " + ", ".join(missing_log_evidence))
        if fallbacks:
            details.append("fallback marker: " + ", ".join(fallbacks))
        raise SystemExit("MXAttention validation failed (" + "; ".join(details) + ")")

    expected = {f"{prompt}-0.mp4": prompt for prompt in prompts}
    missing = [name for name in expected if not (args.video_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing expected videos:\n" + "\n".join(missing))

    probes = {name: probe_video(args.video_dir / name) for name in expected}
    invalid = {
        name: data for name, data in probes.items()
        if (data["width"], data["height"], data["frames"], round(data["fps"])) != (1280, 720, 81, 16)
    }
    if invalid:
        raise SystemExit("Unexpected video metadata: " + json.dumps(invalid, indent=2))

    output_root = args.video_dir.parent
    prompt_map = output_root / "prompts.json"
    prompt_map.write_text(json.dumps(expected, indent=2) + "\n")
    provenance = {
        "config": "mxattention_full",
        "mode": "mxattention_full",
        "qmax": 7.25,
        "hadamard": True,
        "generation_cuda_devices": "1,2",
        "evaluation_cuda_device": "1",
        "seed": 42,
        "prompt_file": str(args.prompt_file.resolve()),
        "videos": probes,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated {len(expected)} MXAttention-full videos")
    print(f"Prompt mapping: {prompt_map}")


if __name__ == "__main__":
    main()
