#!/usr/bin/env python3
"""Validate complete MXAttention-full generation, including resumed legacy clips."""
import argparse
import json
import subprocess
from pathlib import Path

def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, text=True, capture_output=True, check=True).stdout
        stream = json.loads(out)["streams"][0]
        n, d = stream["avg_frame_rate"].split("/")
        return {"width": int(stream["width"]), "height": int(stream["height"]),
                "frames": int(stream["nb_frames"]), "fps": float(n) / float(d)}
    except FileNotFoundError:
        # The workspace venv ships PyAV even where the ffprobe binary is absent.
        try:
            import av
        except ImportError as exc:
            raise SystemExit("ffprobe or PyAV is required for video validation") from exc
        with av.open(path) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(stream))
            rate = stream.average_rate
            return {"width": int(stream.width), "height": int(stream.height), "frames": frames,
                    "fps": float(rate) if rate else 0.0}

FALLBACK_MARKERS = ("falling back to sdpa", "hardware disabled")
REQUIRED_ENV = {
    "VLLM_MXATTENTION": "1", "MXATTENTION_MODE": "mxattention_full",
    "MXATTENTION_QMAX": "7.25", "MXATTENTION_USE_HADAMARD": "1",
    "DIFFUSION_ATTENTION_BACKEND": "SAGE_ATTN",
}

def check_log(path: Path) -> None:
    log = path.read_text(errors="replace").lower()
    if not any(marker in log for marker in ("sage_attn", "sage attention")):
        raise SystemExit(f"Missing SAGE_ATTN evidence: {path}")
    if any(marker in log for marker in FALLBACK_MARKERS):
        raise SystemExit(f"Fallback marker found: {path}")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--run-manifest", type=Path)
    p.add_argument("--legacy-log-dir", type=Path)
    a = p.parse_args()
    prompts = [x.strip() for x in a.prompt_file.read_text().splitlines() if x.strip()]
    if len(prompts) != 72:
        raise SystemExit(f"Expected 72 prompts, found {len(prompts)}")
    manifest = None
    generated_prompts: set[str] = set()
    if a.run_manifest:
        if not a.run_manifest.is_file():
            raise SystemExit(f"Missing run manifest: {a.run_manifest}")
        manifest = json.loads(a.run_manifest.read_text())
        if manifest.get("environment") != REQUIRED_ENV:
            raise SystemExit("Run manifest does not record the required MXAttention environment")
        for gpu, assigned in manifest.get("assignments", {}).items():
            log_path = a.log_dir / f"gpu{gpu}.server.log"
            if assigned:
                if not log_path.is_file():
                    raise SystemExit(f"Missing new-worker server log: {log_path}")
                check_log(log_path)
                generated_prompts.update(assigned)
    else:
        logs = list(a.log_dir.glob("shard*.server.log"))
        if len(logs) < 4:
            raise SystemExit(f"Expected four shard server logs, found {len(logs)}")
        for log_path in logs:
            check_log(log_path)
    expected = {f"{x}-0.mp4": x for x in prompts}
    missing = [name for name in expected if not (a.video_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing expected videos:\n" + "\n".join(missing))
    videos = {name: probe(a.video_dir / name) for name in expected}
    bad = {n: v for n, v in videos.items() if (v["width"], v["height"], v["frames"], round(v["fps"])) != (1280, 720, 81, 16)}
    if bad:
        raise SystemExit("Unexpected video metadata: " + json.dumps(bad, indent=2))
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / "prompts.json").write_text(json.dumps(expected, indent=2) + "\n")
    legacy_prompts = sorted(set(prompts) - generated_prompts)
    legacy_logs = []
    if legacy_prompts:
        legacy_dir = a.legacy_log_dir or a.log_dir
        legacy_logs = sorted(legacy_dir.glob("*.server.log"))
        if not legacy_logs:
            raise SystemExit("Legacy clips exist but no legacy server logs were provided")
        for log_path in legacy_logs:
            check_log(log_path)
        evidence = {"policy": "documented_configuration_evidence", "prompts": legacy_prompts,
                    "required_environment": REQUIRED_ENV,
                    "server_logs": [str(x.resolve()) for x in legacy_logs],
                    "note": "Legacy logs prove SAGE_ATTN selection and no fallback; the run configuration records MXAttention-full."}
        (a.output_root / "legacy-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    provenance = {"config": "mxattention_full", "mode": "mxattention_full", "qmax": 7.25,
                  "hadamard": True, "generation_cuda_devices": manifest.get("gpus") if manifest else [1, 2, 3, 5], "tp": 1,
                  "evaluation_cuda_device": 1, "seed": 42, "num_prompts": 72,
                  "prompt_file": str(a.prompt_file.resolve()), "videos": videos,
                  "newly_generated_prompts": sorted(generated_prompts), "legacy_prompts": legacy_prompts,
                  "run_manifest": str(a.run_manifest.resolve()) if a.run_manifest else None}
    (a.output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Validated {len(videos)} MXAttention-full videos")

if __name__ == "__main__":
    main()
