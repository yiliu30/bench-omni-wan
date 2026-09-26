#!/usr/bin/env python3
"""Resumable paired VBench-I2V Lightning generation on four XPU workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zipfile

import cv2

from run import load_env


UPSTREAM_REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
UPSTREAM_METADATA_SHA256 = "8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8"
OFFICIAL_ARCHIVE_ID = "1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ"
OFFICIAL_FOLDER_ID = "1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx"
FALLBACK_BLOCKS = "0,33,34,38"
ALL_BLOCKS = ",".join(map(str, range(40)))
RECIPES = ("bf16_flash", "mxfp4_sage_flash4")
NO_FALLBACK_RECIPE = "mxfp4_sage_no_fallback"
PURE_RECIPE = "mxfp4_sage_v3_pure"
STANDALONE_RECIPES = (NO_FALLBACK_RECIPE, PURE_RECIPE)
RECIPE_FIELDS = frozenset((
    "RECIPE", "QUANTIZATION", "ATTENTION_BACKEND", "SAGE_ATTN3_XPU_VARIANT",
    "SAGE_ATTN_FORCE_FALLBACK", "SAGE_ATTN_FALLBACK",
    "SAGE_ATTN_FORCE_SDPA_BLOCKS", "SAGE_ATTN_REPORT_FALLBACKS",
    "VLLM_OMNI_XPU_STAGE_WAN_WEIGHTS",
))
RECIPE_ENV_FILES = {
    "bf16_flash": Path(__file__).with_name("i2v_lightning_bf16.env"),
    "mxfp4_sage_flash4": Path(__file__).with_name("i2v_lightning_sage.env"),
    NO_FALLBACK_RECIPE: Path(__file__).with_name("i2v_lightning_sage_no_fallback.env"),
    PURE_RECIPE: Path(__file__).with_name("i2v_lightning_sage_pure.env"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_rows(metadata: Path, count: int) -> list[dict]:
    rows = json.loads(metadata.read_text(encoding="utf-8"))
    selected = [row for row in rows if "imaging_quality" in row["dimension"]][:count]
    if len(selected) != count or len({row["image_name"] for row in selected}) != count:
        raise ValueError(f"Expected {count} distinct imaging_quality images; found {len(selected)} entries")
    if any(Path(row["image_name"]).name != row["image_name"] for row in selected):
        raise ValueError("Unsafe image name in VBench metadata")
    return selected


def config(path: Path) -> dict[str, str]:
    settings, exported = load_env(str(path))
    if exported:
        raise ValueError(f"Unexpected [env] settings in {path}; recipe settings must be explicit")
    recipe = settings.get("RECIPE")
    if recipe is not None and recipe not in STANDALONE_RECIPES:
        raise ValueError("Pass i2v_lightning_50.env to the paired runner; recipe envs load automatically")
    cfg = {key: value for key, value in settings.items() if key not in RECIPE_FIELDS}
    if recipe in STANDALONE_RECIPES:
        baseline, _ = load_env(str(Path(__file__).with_name("i2v_lightning_50.env")))
        forced = "" if recipe == NO_FALLBACK_RECIPE else FALLBACK_BLOCKS
        if (set(cfg) != set(baseline) | {"FALLBACK_BLOCKS"}
                or any(cfg[key] != baseline[key] for key in baseline if key != "OUTPUT")
                or cfg.get("FALLBACK_BLOCKS") != forced
                or cfg["OUTPUT"] == baseline["OUTPUT"]):
            raise ValueError(f"{recipe} run must match the paired campaign except OUTPUT and forced blocks")
    required = ("MODEL", "METADATA", "ARCHIVE", "CROPS", "OUTPUT", "VLLM_OMNI_ROOT",
                "DEEPKLOX_REPO", "PYTHON", "DEVICES", "PORT_BASE")
    for key in required:
        if not cfg.get(key):
            raise ValueError(f"Missing {key} in {path}")
    devices = [int(item) for item in cfg["DEVICES"].split(",")]
    if devices != [0, 1, 2, 3]:
        raise ValueError("This campaign requires exactly physical XPUs 0,1,2,3")
    if (int(cfg["WIDTH"]), int(cfg["HEIGHT"]), int(cfg["NUM_FRAMES"]),
        int(cfg["FPS"]), int(cfg["STEPS"]), int(cfg["SEED"])) != (640, 640, 81, 16, 4, 0):
        raise ValueError("Lightning pair settings must remain 640x640, 81f/16fps, 4 steps, seed 0")
    if cfg.get("VAE_FAST_DECODE") != "0":
        raise ValueError("This campaign uses upstream Wan VAE decode on all four XPUs")
    for selected in (recipe,) if recipe else RECIPES:
        recipe_config(cfg, selected)
    return cfg


def recipe_config(cfg: dict[str, str], recipe: str) -> dict[str, str]:
    path = RECIPE_ENV_FILES[recipe]
    settings, exported = load_env(str(path))
    expected = {
        "RECIPE": recipe,
        "QUANTIZATION": "none" if recipe == "bf16_flash" else "mxfp4",
        "ATTENTION_BACKEND": "SAGE_ATTN_3",
        "SAGE_ATTN3_XPU_VARIANT": "pure_mxfp4" if recipe == PURE_RECIPE else "hybrid",
        "SAGE_ATTN_FORCE_FALLBACK": "flash",
        "SAGE_ATTN_FALLBACK": "sdpa",
        "SAGE_ATTN_FORCE_SDPA_BLOCKS": (
            ALL_BLOCKS if recipe == "bf16_flash" else
            "" if recipe == NO_FALLBACK_RECIPE else FALLBACK_BLOCKS
        ),
        "SAGE_ATTN_REPORT_FALLBACKS": "1",
        "VLLM_OMNI_XPU_STAGE_WAN_WEIGHTS": "1",
    }
    if exported or set(settings) != set(cfg) | set(expected):
        raise ValueError(f"Unexpected or missing recipe configuration in {path}")
    for key, value in {**cfg, **expected}.items():
        if settings[key] != value:
            raise ValueError(f"{path}: {key} differs from the paired campaign: {settings[key]!r}")
    return settings


def extract_official(cfg: dict[str, str]) -> None:
    archive = Path(cfg["ARCHIVE"])
    if digest(Path(cfg["METADATA"])) != UPSTREAM_METADATA_SHA256:
        raise ValueError("Metadata does not match the pinned upstream VBench-I2V revision")
    if not archive.is_file():
        raise FileNotFoundError(
            f"Official crop.zip unavailable: {archive}. Obtain Google Drive file {OFFICIAL_ARCHIVE_ID}; "
            "do not use reconstructed inputs."
        )
    rows = selected_rows(Path(cfg["METADATA"]), int(cfg["COUNT"]))
    names = {row["image_name"] for row in rows}
    output = Path(cfg["CROPS"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite official crops: {output}")
    with zipfile.ZipFile(archive) as zipped:
        matching = {}
        for entry in zipped.infolist():
            parts = PurePosixPath(entry.filename).parts
            if entry.is_dir() or len(parts) < 3 or parts[-2] != "1-1" or parts[-1] not in names:
                continue
            name = parts[-1]
            if name in matching:
                raise ValueError(f"Duplicate official crop: {name}")
            matching[name] = entry
        if set(matching) != names:
            raise ValueError(f"Official archive missing {len(names - set(matching))} selected crops: "
                             f"{sorted(names - set(matching))[:5]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vbench-official-", dir=output.parent) as staged:
            for name, entry in matching.items():
                target = Path(staged) / name
                with zipped.open(entry) as source, target.open("xb") as sink:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        sink.write(chunk)
            Path(staged).replace(output)
    write_json(output.parent / "source.json", {
        "source": f"https://drive.google.com/uc?id={OFFICIAL_ARCHIVE_ID}",
        "archive_sha256": digest(archive),
        "metadata_sha256": digest(Path(cfg["METADATA"])),
        "upstream_commit": UPSTREAM_REVISION,
        "count": len(matching),
        "crop_sha256": {name: digest(output / name) for name in sorted(matching)},
    })


def prepare(cfg: dict[str, str]) -> dict:
    metadata = Path(cfg["METADATA"])
    if digest(metadata) != UPSTREAM_METADATA_SHA256:
        raise ValueError("Metadata does not match the pinned upstream VBench-I2V revision")
    rows = selected_rows(metadata, int(cfg["COUNT"]))
    crops = Path(cfg["CROPS"])
    provenance_path = crops.parent / "source.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Official-crop provenance missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    folder_source = f"https://drive.google.com/drive/folders/{OFFICIAL_FOLDER_ID}"
    archive_source = f"https://drive.google.com/uc?id={OFFICIAL_ARCHIVE_ID}"
    if (provenance.get("source") not in (archive_source, folder_source)
            or provenance.get("metadata_sha256") != digest(metadata)
            or provenance.get("upstream_commit") != UPSTREAM_REVISION):
        raise ValueError("Official crop/metadata provenance mismatch")
    if provenance["source"] == folder_source:
        if provenance.get("count") != len(rows) or set(provenance.get("file_ids", {})) != {
            row["image_name"] for row in rows
        } or any(not file_id for file_id in provenance["file_ids"].values()):
            raise ValueError("Official Drive crop file IDs do not cover the selected inputs")
    missing = [row["image_name"] for row in rows if not (crops / row["image_name"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} official crops missing, e.g. {missing[:5]}")
    items = []
    for index, row in enumerate(rows):
        image = crops / row["image_name"]
        checksum = digest(image)
        if provenance["crop_sha256"].get(row["image_name"]) != checksum:
            raise ValueError(f"Crop checksum mismatch: {image}")
        decoded = cv2.imread(str(image))
        if decoded is None or decoded.shape[0] != decoded.shape[1]:
            raise ValueError(f"Expected decodable square official crop: {image}")
        items.append({"index": index, "prompt": row["prompt_en"], "image_name": row["image_name"],
                      "image_type": row["image_type"], "image": str(image), "sha256": checksum,
                      "device": index % 4, "stem": f"{index:03}"})
    manifest = {"metadata_sha256": digest(metadata), "upstream_commit": UPSTREAM_REVISION,
                "official_source": provenance["source"],
                "official_file_ids": provenance.get("file_ids"),
                "archive_sha256": provenance.get("archive_sha256"),
                "model": cfg["MODEL"], "devices": [0, 1, 2, 3], "width": 640, "height": 640,
                "frames": 81, "fps": 16, "steps": 4, "seed": 0, "guidance": 1.0,
                "boundary_ratio": 0.9, "flow_shift": 1.0, "solver": "euler",
                "fallback_blocks": cfg.get("FALLBACK_BLOCKS", FALLBACK_BLOCKS), "items": items}
    output = Path(cfg["OUTPUT"]) / "manifest.json"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Manifest changed; use a new OUTPUT directory instead of overwriting {output}")
    else:
        write_json(output, manifest)
    return manifest


def revision(path: str) -> dict[str, str]:
    result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True)
    diff = subprocess.run(["git", "-C", path, "diff", "--binary", "HEAD"], check=True,
                          capture_output=True)
    return {"commit": result.stdout.strip(),
            "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest()}


def gpu_preflight(cfg: dict[str, str]) -> None:
    model = Path(cfg["MODEL"])
    if not (model / "model_index.json").is_file():
        raise FileNotFoundError(f"Lightning model not mounted: {model}")
    if not (Path(cfg["VLLM_OMNI_ROOT"]) / "vllm_omni/quantization/mxfp4_xpu.py").is_file():
        raise FileNotFoundError("Validated portable XPU MXFP4 quantization module is missing")
    import torch
    from deepklox import _C
    if not hasattr(_C, "sageattn_v3_hybrid"):
        raise RuntimeError("Sage V3 Hybrid XPU kernel unavailable")
    if torch.xpu.device_count() != 4:
        raise RuntimeError(f"Expected exactly four exposed XPUs; got {torch.xpu.device_count()}")
    identities = []
    for index in range(4):
        props = torch.xpu.get_device_properties(index)
        identities.append(str(props.uuid))
        if torch.xpu.mem_get_info(index)[0] < 70 * 2**30:
            raise RuntimeError(f"XPU {index} has less than 70 GiB free")
    if len(set(identities)) != 4:
        raise RuntimeError(f"Nonunique XPU identities: {identities}")
    print("Verified physical XPU UUIDs:", identities, flush=True)


def validate_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (640, 640, 3):
                raise ValueError(f"Unexpected frame shape in {path}: {frame.shape}")
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != 81 or abs(fps - 16) > 0.05:
        raise ValueError(f"Expected 81 frames at 16fps in {path}; got {len(frames)} at {fps}")
    sampled = frames[::20]
    if any(float(frame.mean()) < 5 or float(frame.std()) < 8 for frame in sampled):
        raise ValueError(f"Black/flat sampled frame in {path}")
    motion = float(cv2.absdiff(frames[0], frames[-1]).mean())
    if motion < 0.1:
        raise ValueError(f"No decoded motion in {path}")
    return {"frames": len(frames), "fps": fps, "motion_mae": motion, "sha256": digest(path)}


def request_json(url: str, data: bytes | None = None, content_type: str | None = None,
                 timeout: int = 90) -> dict:
    headers = {"Content-Type": content_type} if content_type else {}
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url}: HTTP {exc.code}: {exc.read()[:1000]!r}") from exc


def video_request(item: dict) -> tuple[bytes, str]:
    boundary = f"----I2VSubset{uuid.uuid4().hex}"
    fields = {"prompt": item["prompt"], "width": "640", "height": "640", "num_frames": "81",
              "fps": "16", "num_inference_steps": "4", "guidance_scale": "1.0",
              "boundary_ratio": "0.9", "flow_shift": "1.0", "seed": "0",
              "negative_prompt": "", "extra_params": json.dumps({"sample_solver": "euler"})}
    chunks = []
    for key, value in fields.items():
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                       f"{value}\r\n").encode("utf-8"))
    chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; "
                   f"filename=\"{Path(item['image']).name}\"\r\nContent-Type: image/jpeg\r\n\r\n").encode())
    chunks.extend((Path(item["image"]).read_bytes(), b"\r\n", f"--{boundary}--\r\n".encode()))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def generate(url: str, item: dict, output: Path) -> dict:
    body, content_type = video_request(item)
    job = request_json(f"{url}/v1/videos", body, content_type)
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"Missing video job id: {job}")
    deadline = time.monotonic() + 1800
    while job.get("status") not in ("completed", "failed"):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Video job {job_id} exceeded 1800 seconds")
        time.sleep(3)
        job = request_json(f"{url}/v1/videos/{job_id}")
    if job["status"] != "completed":
        raise RuntimeError(f"Video job failed: {job}")
    request = urllib.request.Request(f"{url}/v1/videos/{job_id}/content")
    with urllib.request.urlopen(request, timeout=180) as response, output.open("xb") as sink:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            sink.write(chunk)
    return job


def server_env(cfg: dict[str, str], device: int, recipe: str) -> dict[str, str]:
    env = os.environ.copy()
    settings = recipe_config(cfg, recipe)
    env.update({"ZE_FLAT_DEVICE_HIERARCHY": "FLAT", "ZE_AFFINITY_MASK": str(device),
                "VLLM_OMNI_WAN_VAE_FAST_DECODE": settings["VAE_FAST_DECODE"],
                "DIFFUSION_ATTENTION_BACKEND": settings["ATTENTION_BACKEND"],
                "SAGE_ATTN3_XPU_VARIANT": settings["SAGE_ATTN3_XPU_VARIANT"],
                "SAGE_ATTN_FORCE_FALLBACK": settings["SAGE_ATTN_FORCE_FALLBACK"],
                "SAGE_ATTN_FALLBACK": settings["SAGE_ATTN_FALLBACK"],
                "SAGE_ATTN_FORCE_SDPA_BLOCKS": settings["SAGE_ATTN_FORCE_SDPA_BLOCKS"],
                "SAGE_ATTN_REPORT_FALLBACKS": settings["SAGE_ATTN_REPORT_FALLBACKS"],
                "VLLM_OMNI_XPU_STAGE_WAN_WEIGHTS": settings["VLLM_OMNI_XPU_STAGE_WAN_WEIGHTS"],
                "PYTHONPATH": f"{cfg['DEEPKLOX_REPO']}:{cfg['VLLM_OMNI_ROOT']}"
                + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")})
    return env


def server_command(cfg: dict[str, str], device: int, recipe: str) -> list[str]:
    command = [cfg["PYTHON"], "-m", "vllm_omni.entrypoints.cli.main", "serve", cfg["MODEL"],
               "--omni", "--host", "127.0.0.1", "--port", str(int(cfg["PORT_BASE"]) + device),
               "--tensor-parallel-size", "1", "--enforce-eager", "--init-timeout",
               cfg["INIT_TIMEOUT"], "--stage-init-timeout", cfg["INIT_TIMEOUT"]]
    quantization = recipe_config(cfg, recipe)["QUANTIZATION"]
    if quantization != "none":
        command += ["--quantization", quantization]
    return command


def verify_routing(log_path: Path, recipe: str) -> None:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0" not in log:
        raise RuntimeError(f"Upstream VAE decode not verified in {log_path}")
    reports = re.findall(r"Sage attention V3 call counts: ([^\n\r]+)", log)
    if not reports:
        raise RuntimeError(f"Actual attention call counters missing from {log_path}")
    values = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", reports[-1])}
    if (values.get("cross_sdpa", 0) <= 0 or values.get("nonfinite_sdpa", -1) != 0
            or values.get("sdpa_fallback", -1) != 0):
        raise RuntimeError(f"Unexpected attention routing in {log_path}: {values}")
    sage = values.get("sage", -1)
    forced = values.get("forced_sdpa", -1)
    flash = values.get("fallback_flash", -1)
    if ((recipe == "bf16_flash" and (sage != 0 or forced <= 0 or flash <= 0))
            or (recipe in ("mxfp4_sage_flash4", PURE_RECIPE) and (sage <= 0 or forced <= 0 or flash <= 0))
            or (recipe == NO_FALLBACK_RECIPE and (sage <= 0 or forced != 0 or flash != 0))):
        raise RuntimeError(f"Unexpected Sage routing in {log_path}: {values}")


def serve_and_run(cfg: dict[str, str], manifest: dict, device: int, recipe: str,
                  retry_failed: bool = False) -> None:
    root = Path(cfg["OUTPUT"]) / recipe
    root.mkdir(parents=True, exist_ok=True)
    port = int(cfg["PORT_BASE"]) + device
    url = f"http://127.0.0.1:{port}"
    log_path = root / f"server_xpu{device}.log"
    assigned = [item for item in manifest["items"] if item["device"] == device]
    if assigned and log_path.exists() and all(
        (root / f"{item['stem']}.json").is_file()
        and (record := json.loads((root / f"{item['stem']}.json").read_text(encoding="utf-8"))).get("status") == "completed"
        and record.get("index") == item["index"]
        and record.get("image_sha256") == item["sha256"]
        and record.get("prompt") == item["prompt"]
        and record.get("recipe") == recipe
        and record.get("device") == device
        and record.get("metadata_sha256") == manifest["metadata_sha256"]
        and record.get("video") == validate_video(root / f"{item['stem']}.mp4")
        for item in assigned
    ):
        verify_routing(log_path, recipe)
        return
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use; refusing to connect to another server") from exc
    if log_path.exists():
        log_path.replace(root / f"server_xpu{device}.attempt_{uuid.uuid4().hex}.log")
    with log_path.open("w", encoding="utf-8") as log:
        env = server_env(cfg, device, recipe)
        command = server_command(cfg, device, recipe)
        log.write(f"\nStarting {recipe} XPU {device}: {command}\n")
        log.flush()
        server = subprocess.Popen(command, cwd=cfg["VLLM_OMNI_ROOT"], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + int(cfg["INIT_TIMEOUT"]) + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
                try:
                    with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(3)
            else:
                raise TimeoutError(f"Server startup timed out; see {log_path}")
            server_log = log_path.read_text(encoding="utf-8", errors="replace")
            settings = recipe_config(cfg, recipe)
            forced = settings["SAGE_ATTN_FORCE_SDPA_BLOCKS"]
            forced_log = "[" + ", ".join(forced.split(",")) + "]" if forced else ""
            forced_verified = (f"transformer blocks: {forced_log}" in server_log if forced else
                               "fallback for transformer blocks:" not in server_log)
            if (not forced_verified
                    or f"SageAttention3 XPU kernel variant: {settings['SAGE_ATTN3_XPU_VARIANT']}"
                    not in server_log
                    or (recipe != "bf16_flash" and (
                        "Building quantization config: mxfp4" not in server_log
                        or "Using XPUMxFp4LinearKernel for MXFP4 GEMM" not in server_log))):
                raise RuntimeError(f"Model routing/quantization not verified in {log_path}")
            for item in manifest["items"]:
                if item["device"] != device:
                    continue
                if digest(Path(item["image"])) != item["sha256"]:
                    raise ValueError(f"Official input changed since manifest creation: {item['image']}")
                video = root / f"{item['stem']}.mp4"
                record = root / f"{item['stem']}.json"
                expected = {"index": item["index"], "image_sha256": item["sha256"],
                            "prompt": item["prompt"], "recipe": recipe, "device": device,
                            "metadata_sha256": manifest["metadata_sha256"]}
                if record.exists():
                    saved = json.loads(record.read_text(encoding="utf-8"))
                    if saved.get("status") == "completed" and all(saved.get(k) == v for k, v in expected.items()):
                        if saved.get("video") == validate_video(video):
                            continue
                    if not retry_failed or saved.get("status") != "failed":
                        raise ValueError(f"Existing output invalid; inspect or retry failed item: {record}")
                    if any(saved.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Failed record does not match manifest: {record}")
                    record.replace(record.with_name(f"{record.stem}.failed_{uuid.uuid4().hex}.json"))
                if video.exists():
                    raise FileExistsError(f"Untracked/partial video exists: {video}")
                partial = root / f"{item['stem']}.partial.mp4"
                if partial.exists():
                    if not retry_failed:
                        raise FileExistsError(f"Previous partial video exists: {partial}")
                    partial.replace(partial.with_name(f"{item['stem']}.failed_{uuid.uuid4().hex}.mp4"))
                started = time.time()
                try:
                    job = generate(url, item, partial)
                    video_info = validate_video(partial)
                    partial.replace(video)
                    write_json(record, {**expected, "status": "completed", "video": video_info,
                                        "job_id": job["id"], "duration_seconds": time.time() - started})
                except Exception as exc:
                    write_json(record, {**expected, "status": "failed", "error": repr(exc),
                                        "duration_seconds": time.time() - started})
                    raise
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
    verify_routing(log_path, recipe)


def run(cfg: dict[str, str], manifest: dict, retry_failed: bool = False,
        smoke: bool = False, recipes: tuple[str, ...] = RECIPES) -> None:
    gpu_preflight(cfg)
    manifest_path = Path(cfg["OUTPUT"]) / "manifest.json"
    smoke_path = Path(cfg["OUTPUT"]) / "smoke.json"
    if not smoke and (
        not smoke_path.exists()
        or json.loads(smoke_path.read_text(encoding="utf-8")).get("manifest_sha256") != digest(manifest_path)
    ):
        raise RuntimeError(f"Verified paired smoke run required before campaign: {smoke_path}")
    import torch
    runtime = {
        "vllm_omni": revision(cfg["VLLM_OMNI_ROOT"]),
        "deepklox": revision(cfg["DEEPKLOX_REPO"]),
        "portable_mxfp4_sha256": digest(Path(cfg["VLLM_OMNI_ROOT"]) / "vllm_omni/quantization/mxfp4_xpu.py"),
        "config": {key: value for key, value in cfg.items() if key != "ARCHIVE"},
        "gpu_uuids": [str(torch.xpu.get_device_properties(index).uuid) for index in range(4)],
        "python": sys.version,
        "torch": torch.__version__,
    }
    runtime_path = Path(cfg["OUTPUT"]) / "runtime.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime:
        raise ValueError(f"Runtime changed; start a new OUTPUT directory: {runtime_path}")
    if not runtime_path.exists():
        write_json(runtime_path, runtime)
    recipe_envs = {
        recipe: {"path": str(RECIPE_ENV_FILES[recipe]),
                 "sha256": digest(RECIPE_ENV_FILES[recipe]),
                 "config": recipe_config(cfg, recipe)}
        for recipe in recipes
    }
    recipe_path = Path(cfg["OUTPUT"]) / "recipe_envs.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text(encoding="utf-8")) != recipe_envs:
        raise ValueError(f"Recipe environment changed; start a new OUTPUT directory: {recipe_path}")
    if not recipe_path.exists():
        write_json(recipe_path, recipe_envs)
    active = {**manifest, "items": manifest["items"][:1]} if smoke else manifest
    devices = (0,) if smoke else range(4)
    for recipe in recipes:
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(serve_and_run, cfg, active, device, recipe, retry_failed)
                       for device in devices]
            for future in futures:
                future.result()
    if recipes == RECIPES:
        from compare_i2v_lightning_subset import compare

        summary = compare(Path(cfg["OUTPUT"]), items=active["items"])
        if summary["pairs"] != (1 if smoke else int(cfg["COUNT"])):
            raise RuntimeError(f"Expected {cfg['COUNT']} paired videos, got {summary['pairs']}")
    else:
        recipe = recipes[0]
        root = Path(cfg["OUTPUT"]) / recipe
        videos = sorted(root.glob("[0-9][0-9][0-9].mp4"))
        if len(videos) != len(active["items"]):
            raise RuntimeError(f"Expected {len(active['items'])} {recipe} videos, got {len(videos)}")
        for item in active["items"]:
            record = json.loads((root / f"{item['stem']}.json").read_text(encoding="utf-8"))
            if (record.get("status") != "completed" or record.get("recipe") != recipe
                    or record.get("device") != item["device"]
                    or record.get("image_sha256") != item["sha256"]
                    or record.get("prompt") != item["prompt"]
                    or record.get("video") != validate_video(root / f"{item['stem']}.mp4")):
                raise RuntimeError(f"Invalid {recipe} video record: {item['stem']}")
        if not smoke:
            forced = recipe_config(cfg, recipe)["SAGE_ATTN_FORCE_SDPA_BLOCKS"]
            write_json(Path(cfg["OUTPUT"]) / "summary.json",
                       {"recipe": recipe, "videos": len(videos), "frames_per_video": 81,
                        "forced_fallback_blocks": [int(block) for block in forced.split(",")
                                                   if block.strip()],
                        "vbench_scored": False})
    if smoke:
        details = {"manifest_sha256": digest(manifest_path), "paired_items": 1,
                   "bf16_log": str(Path(cfg["OUTPUT"]) / "bf16_flash/server_xpu0.log"),
                   "mxfp4_log": str(Path(cfg["OUTPUT"]) / "mxfp4_sage_flash4/server_xpu0.log")}
        if recipes != RECIPES:
            details = {"manifest_sha256": digest(manifest_path), "recipe": recipes[0],
                       "verified_items": 1}
        write_json(smoke_path, details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract-official", "prepare", "preflight", "dry-run", "smoke", "run"))
    parser.add_argument("env_file", nargs="?", default="i2v_lightning_50.env")
    parser.add_argument("--retry-failed", action="store_true", help="Archive failed records/partials before retry")
    args = parser.parse_args()
    path = Path(args.env_file) if Path(args.env_file).is_absolute() else Path(__file__).parent / args.env_file
    selected, _ = load_env(str(path))
    cfg = config(path)
    recipes = ((selected["RECIPE"],) if selected.get("RECIPE") in STANDALONE_RECIPES else RECIPES)
    if args.action == "extract-official":
        extract_official(cfg)
    elif args.action == "dry-run":
        selected = selected_rows(Path(cfg["METADATA"]), int(cfg["COUNT"]))
        print(json.dumps({"metadata_sha256": digest(Path(cfg["METADATA"])),
                          "devices": {device: [i for i in range(len(selected)) if i % 4 == device]
                                      for device in range(4)},
                          "servers": {recipe: [server_command(cfg, device, recipe) for device in range(4)]
                                      for recipe in recipes}}, indent=2))
    elif args.action == "preflight":
        prepare(cfg)
        gpu_preflight(cfg)
    elif args.action == "prepare":
        prepare(cfg)
    else:
        run(cfg, prepare(cfg), retry_failed=args.retry_failed, smoke=args.action == "smoke",
            recipes=recipes)


if __name__ == "__main__":
    main()
