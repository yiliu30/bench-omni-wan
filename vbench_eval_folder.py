#!/usr/bin/env python3
"""All-in-one VBench evaluation: run eval on a video folder and report scores + overall.

Usage:
    python vbench_eval_folder.py /path/to/video_folder [--seed 42] [--cuda 2,3]

Outputs per-dimension scores, quality/semantic/total VBench scores,
and optionally delta vs a baseline JSON.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# VBench scoring
sys.path.insert(0, "/home/yiliu7/workspace/VBench/scripts")
from constant import DIM_WEIGHT, NORMALIZE_DIC, QUALITY_WEIGHT, SEMANTIC_WEIGHT


DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
    "temporal_style",
]

SHORT_NAMES = {
    "subject_consistency": "subj_con",
    "background_consistency": "bg_con",
    "temporal_flickering": "temp_flk",
    "motion_smoothness": "mot_smo",
    "aesthetic_quality": "aesth_q",
    "imaging_quality": "img_q",
    "overall_consistency": "overall_con",
    "temporal_style": "temp_style",
}

PYTHON_BIN = "/home/yiliu7/workspace/venvs/omni/bin/python"
EVAL_SCRIPT = str(Path(__file__).parent / "vbench_eval.py")


def run_eval(video_dir: str, output_dir: str, seed: int, cuda: str):
    """Run VBench evaluation using the seeded eval script."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_VISIBLE_DEVICES"] = cuda

    cmd = [
        PYTHON_BIN, EVAL_SCRIPT,
        "--seed", str(seed),
        "--",
        "--videos_path", video_dir,
        "--dimension", *DIMS,
        "--mode=custom_input",
        "--output_path", output_dir,
    ]

    print(f"Running VBench eval on: {video_dir}")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}, CUDA: {cuda}")
    print("-" * 60)

    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: eval failed with return code {result.returncode}")
        sys.exit(1)


def load_scores(output_dir: str) -> dict[str, float]:
    """Load per-dimension scores from eval results JSON."""
    out_path = Path(output_dir)
    results = sorted(out_path.glob("*_eval_results.json"))
    if not results:
        print(f"ERROR: no eval results found in {output_dir}")
        sys.exit(1)

    data = json.loads(results[-1].read_text())
    scores = {}
    for dim in DIMS:
        if dim not in data:
            continue
        dim_data = data[dim]
        per_video = dim_data[1]
        values = [entry["video_results"] for entry in per_video]
        if values:
            mean = sum(values) / len(values)
            if dim == "imaging_quality" and mean > 1.0:
                mean = mean / 100.0
            scores[dim] = mean
    return scores


def normalize_score(dim_key: str, raw: float) -> float:
    """Apply VBench min-max normalization."""
    if dim_key in NORMALIZE_DIC:
        entry = NORMALIZE_DIC[dim_key]
        lo, hi = entry["Min"], entry["Max"]
        return (raw - lo) / (hi - lo) if hi != lo else 0.0
    return raw


def compute_overall(raw_scores: dict[str, float]) -> dict[str, float]:
    """Compute VBench quality/semantic/total from raw dimension scores."""
    normalized = {}
    for dim, val in raw_scores.items():
        key = dim.replace("_", " ")
        normalized[key] = normalize_score(key, val)

    quality_dims = [
        "subject consistency", "background consistency",
        "temporal flickering", "motion smoothness",
        "aesthetic quality", "imaging quality",
    ]
    semantic_dims = ["temporal style", "overall consistency"]

    q_scores = [normalized[d] for d in quality_dims if d in normalized]
    q_weights = [DIM_WEIGHT[d] for d in quality_dims if d in normalized]
    quality = sum(s * w for s, w in zip(q_scores, q_weights)) / sum(q_weights) if q_weights else 0

    s_scores = [normalized[d] for d in semantic_dims if d in normalized]
    s_weights = [DIM_WEIGHT[d] for d in semantic_dims if d in normalized]
    semantic = sum(s * w for s, w in zip(s_scores, s_weights)) / sum(s_weights) if s_weights else 0

    total = (quality * QUALITY_WEIGHT + semantic * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    return {"quality_score": quality, "semantic_score": semantic, "total_score": total}


def print_table(scores: dict[str, float], overall: dict[str, float],
                baseline_scores: dict[str, float] | None = None,
                baseline_overall: dict[str, float] | None = None):
    """Print results as transposed markdown table (dims as columns)."""
    print("\n## VBench Evaluation Results\n")

    all_cols = [SHORT_NAMES[d] for d in DIMS] + ["quality", "semantic", "**total**"]

    # Header row
    header = "| |" + "".join(f" {c} |" for c in all_cols)
    sep = "|---|" + "".join("------:|" for _ in all_cols)
    print(header)
    print(sep)

    # Score row
    row = "| Score |"
    for dim in DIMS:
        val = scores.get(dim)
        row += f" {val:.4f} |" if val is not None else " N/A |"
    for key in ["quality_score", "semantic_score", "total_score"]:
        row += f" **{overall[key]:.4f}** |" if key == "total_score" else f" {overall[key]:.4f} |"
    print(row)

    # Delta row
    if baseline_scores and baseline_overall:
        row = "| Delta |"
        for dim in DIMS:
            val = scores.get(dim)
            base = baseline_scores.get(dim)
            if val is not None and base is not None and base != 0:
                pct = (val - base) / base * 100
                sign = "+" if pct >= 0 else ""
                row += f" {sign}{pct:.2f}% |"
            else:
                row += " N/A |"
        for key in ["quality_score", "semantic_score", "total_score"]:
            pct = (overall[key] - baseline_overall[key]) / baseline_overall[key] * 100
            sign = "+" if pct >= 0 else ""
            bold = "**" if key == "total_score" else ""
            row += f" {bold}{sign}{pct:.2f}%{bold} |"
        print(row)

    print("")


def main():
    parser = argparse.ArgumentParser(description="All-in-one VBench evaluation")
    parser.add_argument("video_dir", help="Path to folder containing .mp4 videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cuda", default="2,3", help="CUDA_VISIBLE_DEVICES (default: 2,3)")
    parser.add_argument("--baseline", help="Path to baseline summary JSON for delta comparison")
    parser.add_argument("--output-dir", help="Override output directory (default: final-score/<basename>)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just load existing results")
    args = parser.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    basename = os.path.basename(video_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).parent / "final-score" / f"{basename}_custom_input_supported")

    os.makedirs(output_dir, exist_ok=True)

    # Run eval
    if not args.skip_eval:
        run_eval(video_dir, output_dir, args.seed, args.cuda)

    # Load scores
    scores = load_scores(output_dir)
    overall = compute_overall(scores)

    # Load baseline if provided
    baseline_scores = None
    baseline_overall = None
    if args.baseline:
        bl = json.loads(Path(args.baseline).read_text())
        # Support both raw scores dict and summary_common10.json format
        if "scores" in bl:
            # Find bf16 baseline in multi-config summary
            for key in bl["scores"]:
                if "bf16" in key:
                    baseline_scores = bl["scores"][key]
                    baseline_overall = bl["vbench_scores"][key]
                    break
        else:
            baseline_scores = bl
            baseline_overall = compute_overall(baseline_scores)

    # Print results
    print_table(scores, overall, baseline_scores, baseline_overall)

    # Save JSON summary
    summary = {
        "video_dir": video_dir,
        "seed": args.seed,
        "num_videos": len(list(Path(video_dir).glob("*.mp4"))),
        "scores": scores,
        "overall": overall,
    }
    summary_file = Path(output_dir) / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
