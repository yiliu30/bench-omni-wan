#!/usr/bin/env python3
"""Score the three completed Lightning I2V arms with VBench MUSIQ-SPAQ on XPU."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
PAIRED = ROOT / "results/i2v_lightning_50_upstream_vae"
NO_FALLBACK = ROOT / "results/i2v_lightning_50_no_fallback"
OUTPUT = ROOT / "results/i2v_imaging_quality_xpu"
WEIGHTS = Path("/home/yiliu7/vbench_i2v/weights/musiq_spaq_ckpt-358bb6af.pth")
WEIGHTS_SHA256 = "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50"
PACKAGES = "/home/yiliu7/vbench_i2v/scoring_pkgs"
ARMS = {
    "bf16_flash": PAIRED / "bf16_flash",
    "mxfp4_sage_flash4": PAIRED / "mxfp4_sage_flash4",
    "mxfp4_sage_no_fallback": NO_FALLBACK / "mxfp4_sage_no_fallback",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save(path: Path, data: dict) -> None:
    from i2v_lightning_subset import write_json

    write_json(path, data)


def inputs() -> list[dict]:
    paired = json.loads((PAIRED / "manifest.json").read_text(encoding="utf-8"))
    other = json.loads((NO_FALLBACK / "manifest.json").read_text(encoding="utf-8"))
    if ({key: value for key, value in paired.items() if key != "fallback_blocks"}
            != {key: value for key, value in other.items() if key != "fallback_blocks"}
            or other["fallback_blocks"] != "" or len(paired["items"]) != 50):
        raise ValueError("The three scoring arms do not share the same 50 official inputs")
    for item in paired["items"]:
        for arm, directory in ARMS.items():
            video = directory / f"{item['stem']}.mp4"
            record = json.loads((directory / f"{item['stem']}.json").read_text(encoding="utf-8"))
            if (record.get("status") != "completed" or record.get("recipe") != arm
                    or record.get("index") != item["index"]
                    or record.get("device") != item["device"]
                    or record.get("image_sha256") != item["sha256"]
                    or record.get("prompt") != item["prompt"]
                    or record.get("video", {}).get("sha256") != digest(video)
                    or record["video"].get("frames") != 81):
                raise ValueError(f"Unmatched or invalid source video: {video}")
    return paired["items"]


def decode(video: Path, output: Path) -> None:
    import decord
    import numpy as np

    reader = decord.VideoReader(str(video), num_threads=1)
    if len(reader) != 81:
        raise ValueError(f"Expected 81 frames: {video}")
    frames = reader.get_batch(range(len(reader))).asnumpy()
    if frames.shape != (81, 640, 640, 3):
        raise ValueError(f"Wrong decoded shape in {video}: {frames.shape}")
    np.save(output, frames, allow_pickle=False)


def score_frames(model, frames, device) -> list[float]:
    import torch
    from torchvision import transforms

    # Mirrors vbench.imaging_quality.transform(..., preprocess_mode="longer").
    images = torch.Tensor(frames).permute(0, 3, 1, 2)
    images = transforms.Resize((512, 512), antialias=False)(images) / 255.
    with torch.no_grad():
        return [float(model(frame.unsqueeze(0).to(device))) for frame in images]


def worker(index: int, items: list[dict], limit: int | None = None) -> None:
    import numpy as np
    import torch
    from pyiqa.archs.musiq_arch import MUSIQ

    if torch.xpu.device_count() != 1:
        raise RuntimeError(f"Expected one masked XPU for worker {index}, got {torch.xpu.device_count()}")
    device = torch.device("xpu:0")
    model = MUSIQ(pretrained_model_path=str(WEIGHTS)).to(device)
    model.training = False
    assigned = [item for item in items if item["device"] == index]
    if limit is not None:
        assigned = assigned[:limit]
    for item in assigned:
        for arm, directory in ARMS.items():
            video = directory / f"{item['stem']}.mp4"
            record_path = OUTPUT / arm / f"{item['stem']}.json"
            expected = {"recipe": arm, "index": item["index"], "device": index,
                        "image_sha256": item["sha256"], "prompt": item["prompt"],
                        "video_sha256": digest(video), "weights_sha256": WEIGHTS_SHA256,
                        "preprocessing": "VBench longer: RGB float32, 512x512, /255"}
            if record_path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if all(record.get(k) == v for k, v in expected.items()) and len(record.get("frames", [])) == 81:
                    continue
                raise ValueError(f"Scoring record differs from source; inspect {record_path}")
            with tempfile.TemporaryDirectory(prefix="musiq-decode-", dir=OUTPUT) as temporary:
                frame_path = Path(temporary) / "frames.npy"
                subprocess.run([sys.executable, str(Path(__file__)), "decode",
                                str(video), str(frame_path)], check=True)
                frames = np.load(frame_path, allow_pickle=False)
                if frames.shape != (81, 640, 640, 3):
                    raise ValueError(f"Unexpected frames in {video}: {frames.shape}")
                scores = score_frames(model, frames, device)
            if len(scores) != 81 or any(not np.isfinite(score) for score in scores):
                raise ValueError(f"Invalid MUSIQ-SPAQ scores for {video}")
            save(record_path, {**expected, "video_path": str(video), "frames": scores,
                               "video_results": sum(scores) / len(scores)})
            print(f"XPU {index}: {arm} {item['stem']} = {sum(scores) / len(scores):.4f}", flush=True)


def aggregate(items: list[dict], limit: int | None = None) -> dict:
    selected = items[:limit] if limit is not None else items
    rows = []
    summary = {"metric": "VBench imaging_quality MUSIQ-SPAQ", "preprocessing": "longer",
               "frames_per_video": 81, "official_protocol": False, "pairs": len(selected)}
    for arm in ARMS:
        scores = []
        for item in selected:
            record = json.loads((OUTPUT / arm / f"{item['stem']}.json").read_text(encoding="utf-8"))
            if len(record["frames"]) != 81 or record["index"] != item["index"]:
                raise ValueError(f"Missing or invalid scoring record: {arm}/{item['stem']}")
            scores.append(record["video_results"])
            rows.append((item["index"], arm, item["device"], record["video_path"],
                         record["video_sha256"], record["video_results"],
                         record["video_results"] / 100.))
        summary[arm] = {"videos": len(scores), "mean_raw": sum(scores) / len(scores),
                        "vbench_normalized": sum(scores) / len(scores) / 100.}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_video.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pair", "recipe", "xpu", "video_path", "video_sha256",
                         "musiq_spaq_raw", "vbench_normalized"))
        writer.writerows(rows)
    save(OUTPUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "smoke", "decode", "worker"))
    parser.add_argument("extra", nargs="*")
    args = parser.parse_args()
    if args.action == "decode":
        if len(args.extra) != 2:
            parser.error("decode requires video and output paths")
        decode(Path(args.extra[0]), Path(args.extra[1]))
        return
    if digest(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError(f"VBench MUSIQ-SPAQ checkpoint checksum mismatch: {WEIGHTS}")
    items = inputs()
    if args.action == "worker":
        if len(args.extra) != 2:
            parser.error("worker requires device and limit")
        worker(int(args.extra[0]), items, int(args.extra[1]) or None)
        return
    if args.extra:
        parser.error(f"Unexpected arguments: {args.extra}")
    import torch
    if torch.xpu.device_count() != 4:
        raise RuntimeError(f"Expected 4 distinct XPUs, got {torch.xpu.device_count()}")
    uuids = [str(torch.xpu.get_device_properties(i).uuid) for i in range(4)]
    if len(set(uuids)) != 4:
        raise RuntimeError(f"Nonunique XPU identities: {uuids}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {"checkpoint_sha256": WEIGHTS_SHA256, "xpu_uuids": uuids,
                  "paired_manifest_sha256": digest(PAIRED / "manifest.json"),
                  "no_fallback_manifest_sha256": digest(NO_FALLBACK / "manifest.json"),
                  "scorer_sha256": digest(Path(__file__)), "torch": torch.__version__,
                  "pyiqa": "0.1.16", "decord": "0.6.0", "frames_per_video": 81,
                  "scoring_mode": "VBench imaging_quality longer; one video per pair"}
    path = OUTPUT / "provenance.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != provenance:
        raise ValueError(f"Scoring environment changed: {path}")
    if not path.exists():
        save(path, provenance)
    limit = 1 if args.action == "smoke" else 0
    processes = []
    for device in (0,) if limit else range(4):
        environment = os.environ.copy()
        environment["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
        environment["ZE_AFFINITY_MASK"] = str(device)
        environment["PYTHONPATH"] = PACKAGES + (":" + environment["PYTHONPATH"]
                                              if environment.get("PYTHONPATH") else "")
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__)), "worker",
                                           str(device), str(limit)], env=environment))
    errors = [process.wait() for process in processes]
    if any(errors):
        raise RuntimeError(f"XPU scoring workers failed: {errors}")
    print(json.dumps(aggregate(items, limit=limit or None), indent=2))


if __name__ == "__main__":
    main()
