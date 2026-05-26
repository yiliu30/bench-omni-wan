#!/usr/bin/env python3
"""Evaluate all video subfolders in a directory using VBench."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VBENCH_EVAL_SH = Path(__file__).parent / "vbench_eval.sh"
RESULTS_DIR = Path(__file__).parent / "final-score"

DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "temporal_style",
]


def find_video_dirs(root: Path) -> list[Path]:
    """Find all subdirectories containing .mp4 files."""
    dirs = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and any(entry.glob("*.mp4")):
            dirs.append(entry)
    return dirs


def get_result_file(test_name: str) -> Path | None:
    """Find the eval_results JSON for a given test name."""
    out_dir = RESULTS_DIR / f"{test_name}_custom_input_supported"
    if not out_dir.exists():
        return None
    results = sorted(out_dir.glob("*_eval_results.json"))
    return results[-1] if results else None


def run_eval(video_dir: Path, cuda_devices: str, eval_seed: int) -> Path:
    """Run vbench_eval.sh for a single video directory."""
    python_bin = "/home/yiliu7/workspace/venvs/omni/bin/python"
    seeded_eval_py = str(Path(__file__).parent / "vbench_eval.py")
    test_basename = video_dir.name
    test_name = f"{test_basename}_custom_input_supported"
    test_out_dir = RESULTS_DIR / test_name
    test_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(eval_seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    cmd = [
        python_bin, seeded_eval_py,
        "--seed", str(eval_seed),
        "--",
        "--videos_path", str(video_dir),
        "--dimension", *DIMS,
        "--mode", "custom_input",
        "--output_path", str(test_out_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Evaluating: {video_dir.name}")
    print(f"  Videos: {len(list(video_dir.glob('*.mp4')))} mp4 files")
    print(f"  Output: {test_out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: evaluation failed for {video_dir.name} (exit code {result.returncode})")
        return None

    result_file = get_result_file(test_basename)
    if result_file:
        print(f"  Results saved: {result_file}")
    return result_file


def load_scores(result_file: Path) -> dict[str, float]:
    """Load dimension scores from a results JSON."""
    data = json.loads(result_file.read_text())
    scores = {}
    for dim, value in data.items():
        if isinstance(value, list) and len(value) >= 1:
            scores[dim] = value[0]
    return scores


def print_summary(all_scores: dict[str, dict[str, float]]):
    """Print a comparison table of all evaluated directories."""
    if not all_scores:
        print("\nNo results to summarize.")
        return

    dims = DIMS
    name_width = max(len(name) for name in all_scores) + 2

    print(f"\n{'='*80}")
    print("VBENCH EVALUATION SUMMARY")
    print(f"{'='*80}")

    # Header
    header = f"{'Config':<{name_width}}"
    for dim in dims:
        short = dim.replace("_consistency", "_con").replace("_smoothness", "_smo") \
                   .replace("_flickering", "_flk").replace("_quality", "_q") \
                   .replace("temporal_", "t_").replace("overall_", "o_") \
                   .replace("aesthetic", "aesth").replace("imaging", "img")
        header += f" {short:>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for name, scores in sorted(all_scores.items()):
        row = f"{name:<{name_width}}"
        for dim in dims:
            val = scores.get(dim)
            row += f" {val:>10.4f}" if val is not None else f" {'N/A':>10}"
        print(row)

    print(f"{'='*80}")

    # Save summary JSON
    summary_file = RESULTS_DIR / "summary_all.json"
    json_out = {name: scores for name, scores in sorted(all_scores.items())}
    summary_file.write_text(json.dumps(json_out, indent=2))
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run VBench on all video subdirectories.")
    parser.add_argument("--root", type=Path, default=Path("/home/yiliu7/workspace/wan-res"),
                        help="Root directory containing video subdirectories")
    parser.add_argument("--cuda-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "2,3"),
                        help="CUDA devices to use (default: 2,3)")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip directories that already have results")
    parser.add_argument("--dirs", nargs="*",
                        help="Only evaluate these specific subdirectory names")
    args = parser.parse_args()

    video_dirs = find_video_dirs(args.root)
    if args.dirs:
        video_dirs = [d for d in video_dirs if d.name in args.dirs]

    print(f"Found {len(video_dirs)} video directories in {args.root}:")
    for d in video_dirs:
        n_videos = len(list(d.glob("*.mp4")))
        print(f"  {d.name}/ ({n_videos} videos)")

    all_scores = {}

    for video_dir in video_dirs:
        # Check if already evaluated
        existing = get_result_file(video_dir.name)
        if existing and args.skip_existing:
            print(f"\nSkipping {video_dir.name} (results exist: {existing})")
            all_scores[video_dir.name] = load_scores(existing)
            continue

        result_file = run_eval(video_dir, args.cuda_devices, args.seed)
        if result_file:
            all_scores[video_dir.name] = load_scores(result_file)

    # Also load any previously existing results not in this run
    if args.skip_existing:
        for d in find_video_dirs(args.root):
            if d.name not in all_scores:
                existing = get_result_file(d.name)
                if existing:
                    all_scores[d.name] = load_scores(existing)

    print_summary(all_scores)


if __name__ == "__main__":
    main()
