#!/usr/bin/env python3
"""Generate and save videos for accuracy evaluation (e.g. VBench).

Reuses the same .env files as run.py for server config. Uses the synchronous
/v1/videos/sync endpoint so each video is saved as soon as it finishes.
Already-generated videos are skipped (resume-safe).

Usage:
    python generate.py                        # default.env, subject_consistency prompts
    python generate.py mxfp4.env              # mxfp4 config
    python generate.py default.env --no-server --num-prompts 10
    python generate.py default.env --prompt "A cat walking on the beach"
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# VBench prompt sets
# ---------------------------------------------------------------------------

VBENCH_PROMPT_URLS = {
    "subject_consistency": (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/"
        "prompts/prompts_per_dimension/subject_consistency.txt"
    ),
    "scene": (
        "https://raw.githubusercontent.com/Vchitect/VBench/master/"
        "prompts/prompts_per_dimension/scene.txt"
    ),
}

# ---------------------------------------------------------------------------
# Shared helpers (same as run.py)
# ---------------------------------------------------------------------------


def load_env(path: str) -> dict[str, str]:
    cfg = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            cfg[key.strip()] = value.strip()
    return cfg


def bool_val(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


def wait_for_server(host: str, port: int, timeout: int = 600) -> bool:
    import socket

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((host, port)) == 0:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_tree(pid: int) -> None:
    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for ch in children:
            ch.terminate()
        psutil.wait_procs(children, timeout=10)
        parent.terminate()
        parent.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def download_prompts(prompt_set: str, cache_dir: str) -> list[str]:
    url = VBENCH_PROMPT_URLS.get(prompt_set)
    if url is None:
        raise ValueError(
            f"Unknown prompt set: {prompt_set}. "
            f"Available: {list(VBENCH_PROMPT_URLS)}"
        )
    local = os.path.join(cache_dir, f"{prompt_set}.txt")
    if not os.path.exists(local):
        print(f"Downloading {prompt_set} prompts to {local} ...")
        urllib.request.urlretrieve(url, local)
    with open(local) as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Video generation (sync endpoint)
# ---------------------------------------------------------------------------


def generate_video_sync(
    base_url: str,
    prompt: str,
    output_path: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int,
    fps: int,
    seed: int | None = None,
    extra_fields: dict | None = None,
) -> str:
    """POST to /v1/videos/sync, stream response bytes to output_path.

    Returns the inference time from the X-Inference-Time-S header (or "?").
    """
    boundary = "----BenchWanBoundary"
    fields = {
        "prompt": prompt,
        "height": str(height),
        "width": str(width),
        "num_frames": str(num_frames),
        "num_inference_steps": str(num_inference_steps),
        "fps": str(fps),
    }
    if seed is not None:
        fields["seed"] = str(seed)
    if extra_fields:
        fields.update(extra_fields)

    body_parts = []
    for key, value in fields.items():
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{key}"\r\n'.encode()
        )
        body_parts.append(value.encode())
    body_parts.append(f"--{boundary}--".encode())
    body = b"\r\n".join(body_parts)

    req = urllib.request.Request(
        f"{base_url}/v1/videos/sync",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=3600) as resp:
        with open(output_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        return resp.headers.get("X-Inference-Time-S", "?")


# ---------------------------------------------------------------------------
# Server launcher (reuses run.py logic)
# ---------------------------------------------------------------------------


def build_server_cmd(cfg: dict[str, str]) -> list[str]:
    python = cfg.get("PYTHON", sys.executable)
    model = cfg["MODEL"]
    host = cfg.get("HOST", "127.0.0.1")
    port = cfg.get("PORT", "8099")
    tp = cfg.get("TP", "1")

    cmd = [
        python, "-m", "vllm_omni.entrypoints.cli.main",
        "serve", model,
        "--omni",
        "--host", host,
        "--port", port,
        "--tensor-parallel-size", tp,
    ]
    if bool_val(cfg.get("ENFORCE_EAGER", "false")):
        cmd.append("--enforce-eager")
    if bool_val(cfg.get("ENABLE_CPU_OFFLOAD", "false")):
        cmd.append("--enable-cpu-offload")

    extra = cfg.get("EXTRA_SERVE_ARGS", "").strip()
    if extra:
        cmd.extend(extra.split())

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate & save videos for accuracy evaluation"
    )
    parser.add_argument(
        "env_file", nargs="?", default="default.env",
        help="Path to .env config file (default: default.env)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save videos (default: ./output/<env_name>)",
    )
    parser.add_argument(
        "--prompt-set", default="subject_consistency",
        choices=list(VBENCH_PROMPT_URLS),
        help="VBench prompt set to use",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Single prompt (overrides --prompt-set)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompts (default: all)",
    )
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Resolve env file
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = Path(__file__).parent / env_path
    if not env_path.exists():
        print(f"ERROR: Config file not found: {env_path}")
        sys.exit(1)

    cfg = load_env(str(env_path))
    omni_root = Path(cfg.get("VLLM_OMNI_ROOT", ".")).resolve()

    # Output dir: CLI --output-dir > env OUTPUT_DIR > ./output/<env_name>
    env_name = env_path.stem
    output_dir = (
        args.output_dir
        or cfg.get("OUTPUT_DIR")
        or str(Path(__file__).parent / "output" / env_name)
    )
    os.makedirs(output_dir, exist_ok=True)

    # Prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = download_prompts(args.prompt_set, str(Path(__file__).parent))
    if args.num_prompts:
        prompts = prompts[: args.num_prompts]
    elif cfg.get("NUM_PROMPTS"):
        prompts = prompts[: int(cfg["NUM_PROMPTS"])]

    # Read generation params from env
    width = int(cfg.get("WIDTH", "1280"))
    height = int(cfg.get("HEIGHT", "720"))
    num_frames = int(cfg.get("NUM_FRAMES", "81"))
    num_inference_steps = int(cfg.get("NUM_INFERENCE_STEPS", "40"))
    fps = int(cfg.get("FPS", "16"))
    seed_str = cfg.get("SEED", "")
    seed = int(seed_str) if seed_str else None

    host = cfg.get("HOST", "127.0.0.1")
    port = int(cfg.get("PORT", "8099"))
    base_url = f"http://{host}:{port}"

    print("=" * 60)
    print(f"  Generate videos: {env_path.name}")
    print("=" * 60)
    print(f"  Model:      {cfg.get('MODEL')}")
    print(f"  Resolution: {width}x{height}, {num_frames} frames, {fps} fps")
    print(f"  Steps:      {num_inference_steps}")
    print(f"  Seed:       {seed}")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    if args.dry_run:
        for i, p in enumerate(prompts):
            print(f"  [{i+1}] {p[:80]}")
        print("[DRY RUN] Exiting.")
        return

    # Start server if needed
    server_proc = None
    server_log_fh = None
    server_log_path = os.path.join(output_dir, "server.log")
    if not args.no_server:
        # Check for stale process on the port
        import socket as _sock

        with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as _s:
            _s.settimeout(1)
            if _s.connect_ex((host, port)) == 0:
                print(
                    f"ERROR: Port {port} is already in use. "
                    f"Kill the existing server first, or use --no-server."
                )
                sys.exit(1)

        server_cmd = build_server_cmd(cfg)
        print(f"Server cmd: {' '.join(server_cmd)}")

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # Ensure the venv bin is on PATH so JIT tools like ninja are found
        python_bin_dir = str(Path(cfg.get("PYTHON", sys.executable)).resolve().parent)
        env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
        cuda_devices = cfg.get("CUDA_DEVICES", "")
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        server_log_fh = open(server_log_path, "w")
        print(f"Server log: {server_log_path}")

        server_proc = subprocess.Popen(
            server_cmd, env=env, cwd=str(omni_root), preexec_fn=os.setsid,
            stdout=server_log_fh, stderr=subprocess.STDOUT,
        )
        if not wait_for_server(host, port, args.timeout):
            print(f"ERROR: Server failed to start within {args.timeout}s")
            print(f"Check server log: {server_log_path}")
            kill_tree(server_proc.pid)
            server_log_fh.close()
            sys.exit(1)
        print(f"Server ready on {host}:{port}\n")

    try:
        total_start = time.time()
        generated = 0
        skipped = 0

        for i, prompt in enumerate(prompts):
            # Filename: "{prompt}-0.mp4" matching bench_wan_online.py convention
            safe_name = prompt.replace("/", "_")
            output_path = os.path.join(output_dir, f"{safe_name}-0.mp4")

            if os.path.exists(output_path):
                print(f"[{i+1}/{len(prompts)}] SKIP (exists): {prompt[:60]}...")
                skipped += 1
                continue

            print(f"[{i+1}/{len(prompts)}] Generating: {prompt[:60]}...")
            t0 = time.time()
            try:
                inf_time = generate_video_sync(
                    base_url=base_url,
                    prompt=prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    fps=fps,
                    seed=seed,
                )
                elapsed = time.time() - t0
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(
                    f"  Done in {elapsed:.1f}s "
                    f"(inference: {inf_time}s, {size_mb:.1f}MB)"
                )
                generated += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)

        total = time.time() - total_start
        print(f"\nDone. Generated: {generated}, Skipped: {skipped}, "
              f"Total time: {total:.1f}s")
        print(f"Videos saved to: {output_dir}/")

    finally:
        if server_proc is not None:
            print("Shutting down server...")
            kill_tree(server_proc.pid)
        if server_log_fh is not None:
            server_log_fh.close()
            print(f"Server log saved to: {server_log_path}")


if __name__ == "__main__":
    main()
