#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generic diffusion benchmark runner.

Reads all configuration from a single .env file, launches the vLLM-Omni
server, runs the benchmark client, then tears down the server.

Usage:
    python run.py                       # uses default.env
    python run.py smoke.env             # uses smoke.env
    python run.py eager.env --dry-run   # print commands only
    python run.py default.env --no-server --server-only
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Env file loader
# ---------------------------------------------------------------------------

def load_env(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse env file. Returns (cfg, export_env).

    Lines under an [env] section are environment variables to export
    to the server subprocess. All other lines are config values.
    """
    cfg = {}
    export_env = {}
    in_env_section = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                in_env_section = line.lower() == "[env]"
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if in_env_section:
                export_env[key.strip()] = value.strip()
            else:
                cfg[key.strip()] = value.strip()
    return cfg, export_env


def bool_val(s: str) -> bool:
    return s.lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_server(host: str, port: int, timeout: int = 600) -> bool:
    """Block until server accepts TCP connections."""
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
    """Kill process and all children."""
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


def build_server_cmd(cfg: dict[str, str]) -> list[str]:
    """Build the vllm serve command from env config."""
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


def build_bench_cmd(cfg: dict[str, str]) -> list[str]:
    """Build the benchmark client command from env config."""
    python = cfg.get("PYTHON", sys.executable)
    omni_root = Path(cfg.get("VLLM_OMNI_ROOT", ".")).resolve()
    bench_script = cfg.get(
        "BENCHMARK_SCRIPT",
        "benchmarks/diffusion/diffusion_benchmark_serving.py",
    )
    script_path = omni_root / bench_script
    host = cfg.get("HOST", "127.0.0.1")
    port = cfg.get("PORT", "8099")

    cmd = [
        python, str(script_path),
        "--base-url", f"http://{host}:{port}",
        "--model", cfg["MODEL"],
        "--task", cfg.get("TASK", "t2v"),
        "--backend", cfg.get("BACKEND", "v1/videos"),
        "--dataset", cfg.get("DATASET", "vbench"),
        "--num-prompts", cfg.get("NUM_PROMPTS", "5"),
        "--max-concurrency", cfg.get("MAX_CONCURRENCY", "1"),
        "--width", cfg.get("WIDTH", "1280"),
        "--height", cfg.get("HEIGHT", "720"),
        "--num-frames", cfg.get("NUM_FRAMES", "81"),
        "--fps", cfg.get("FPS", "16"),
        "--num-inference-steps", cfg.get("NUM_INFERENCE_STEPS", "40"),
    ]
    seed = cfg.get("SEED", "")
    if seed:
        cmd.extend(["--seed", seed])
    if bool_val(cfg.get("ENABLE_NEGATIVE_PROMPT", "false")):
        cmd.append("--enable-negative-prompt")
    save_dir = cfg.get("SAVE_VIDEOS", "")
    if save_dir:
        cmd.extend(["--save-videos", save_dir])

    return cmd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic diffusion benchmark runner (env-file driven)"
    )
    parser.add_argument(
        "env_file", nargs="?", default="default.env",
        help="Path to .env config file (default: default.env)",
    )
    parser.add_argument("--no-server", action="store_true",
                        help="Don't launch server (connect to existing)")
    parser.add_argument("--server-only", action="store_true",
                        help="Only launch server, don't run benchmark")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Server startup timeout in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve env file relative to this script's directory
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = Path(__file__).parent / env_path
    if not env_path.exists():
        print(f"ERROR: Config file not found: {env_path}")
        sys.exit(1)

    cfg, export_env = load_env(str(env_path))
    omni_root = Path(cfg.get("VLLM_OMNI_ROOT", ".")).resolve()

    # --- Print config summary ---
    print("=" * 60)
    print(f"  Benchmark Config: {env_path.name}")
    print("=" * 60)
    print(f"  Model:           {cfg.get('MODEL')}")
    print(f"  Python:          {cfg.get('PYTHON')}")
    print(f"  TP size:         {cfg.get('TP', '1')}")
    print(f"  Port:            {cfg.get('PORT', '8099')}")
    print(f"  Enforce eager:   {cfg.get('ENFORCE_EAGER', 'false')}")
    print(f"  CPU offload:     {cfg.get('ENABLE_CPU_OFFLOAD', 'false')}")
    print(f"  CUDA devices:    {cfg.get('CUDA_DEVICES', 'all')}")
    print(f"  Extra args:      {cfg.get('EXTRA_SERVE_ARGS', '')}")
    print(f"  vllm-omni root:  {omni_root}")
    print("-" * 60)
    print(f"  Task:            {cfg.get('TASK', 't2v')}")
    print(f"  Resolution:      {cfg.get('WIDTH', '1280')}x{cfg.get('HEIGHT', '720')}")
    print(f"  Frames:          {cfg.get('NUM_FRAMES', '81')}")
    print(f"  Steps:           {cfg.get('NUM_INFERENCE_STEPS', '40')}")
    print(f"  Prompts:         {cfg.get('NUM_PROMPTS', '5')}")
    print(f"  Concurrency:     {cfg.get('MAX_CONCURRENCY', '1')}")
    print("=" * 60)

    server_cmd = build_server_cmd(cfg)
    bench_cmd = build_bench_cmd(cfg)

    print(f"\n  Server command:\n    {' '.join(server_cmd)}\n")
    print(f"  Benchmark command:\n    {' '.join(bench_cmd)}\n")

    if args.dry_run:
        print("[DRY RUN] Exiting without execution.")
        return

    # Validate
    if not omni_root.is_dir():
        print(f"ERROR: vllm-omni root not found: {omni_root}")
        sys.exit(1)

    host = cfg.get("HOST", "127.0.0.1")
    port = int(cfg.get("PORT", "8099"))

    # --- Launch server ---
    server_proc = None
    if not args.no_server:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env.update(export_env)
        cuda_devices = cfg.get("CUDA_DEVICES", "")
        if cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        print("Starting vLLM-Omni server...")
        server_proc = subprocess.Popen(
            server_cmd,
            env=env,
            cwd=str(omni_root),
            preexec_fn=os.setsid,
        )

        if not wait_for_server(host, port, args.timeout):
            print(f"ERROR: Server failed to start within {args.timeout}s")
            kill_tree(server_proc.pid)
            sys.exit(1)

        print(f"Server ready on {host}:{port}")

    if args.server_only:
        print("Server running (--server-only). Press Ctrl+C to stop.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            kill_tree(server_proc.pid)
        return

    # --- Run benchmark ---
    try:
        print("\nRunning benchmark...\n")
        result = subprocess.run(bench_cmd, cwd=str(omni_root))
        print(f"\nBenchmark exited with code {result.returncode}")
    finally:
        if server_proc:
            print("Shutting down server...")
            kill_tree(server_proc.pid)


if __name__ == "__main__":
    main()
