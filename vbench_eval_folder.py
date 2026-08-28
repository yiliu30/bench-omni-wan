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
    # "human_action",  # disabled: timm VisionTransformer incompatibility (cache_dir kwarg)
]

SHORT_NAMES = {
    "subject_consistency": "subj_con",
    "background_consistency": "bg_con",
    "temporal_flickering": "temp_flk",
    "motion_smoothness": "mot_smo",
    "dynamic_degree": "dyn_deg",
    "aesthetic_quality": "aesth_q",
    "imaging_quality": "img_q",
    "overall_consistency": "overall_con",
    "temporal_style": "temp_style",
    "human_action": "human_act",
}

EVAL_SCRIPT = str(Path(__file__).parent / "vbench_eval.py")


def vbench_paths(args: argparse.Namespace) -> tuple[Path, str]:
    vbench_dir = Path(args.vbench_dir or os.environ.get("VBENCH_DIR", "/home/yiliu7/workspace/VBench"))
    python_bin = args.vbench_python or os.environ.get("VBENCH_PYTHON", "/home/yiliu7/workspace/venvs/omni/bin/python")
    if not (vbench_dir / "evaluate.py").is_file():
        raise SystemExit(f"VBench checkout not found: {vbench_dir}")
    if not Path(python_bin).is_file():
        raise SystemExit(f"VBench Python not found: {python_bin}")
    return vbench_dir, python_bin


def vbench_constants(vbench_dir: Path):
    sys.path.insert(0, str(vbench_dir / "scripts"))
    from constant import DIM_WEIGHT, NORMALIZE_DIC, QUALITY_WEIGHT, SEMANTIC_WEIGHT
    return DIM_WEIGHT, NORMALIZE_DIC, QUALITY_WEIGHT, SEMANTIC_WEIGHT


def run_eval(video_dir: str, output_dir: str, seed: int, cuda: str, vbench_dir: Path, python_bin: str, prompt_file: str | None):
    """Run VBench evaluation using the seeded eval script."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["CUDA_VISIBLE_DEVICES"] = cuda
    env["VBENCH_EVALUATE"] = str(vbench_dir / "evaluate.py")

    cmd = [
        python_bin, EVAL_SCRIPT,
        "--seed", str(seed),
        "--",
        "--videos_path", video_dir,
        "--dimension", *DIMS,
        "--mode=custom_input",
        "--output_path", output_dir,
    ]
    if prompt_file:
        cmd.extend(["--prompt_file", prompt_file])

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


def normalize_score(dim_key: str, raw: float, normalize_dic: dict) -> float:
    """Apply VBench min-max normalization."""
    if dim_key in normalize_dic:
        entry = normalize_dic[dim_key]
        lo, hi = entry["Min"], entry["Max"]
        return (raw - lo) / (hi - lo) if hi != lo else 0.0
    return raw


def compute_overall(raw_scores: dict[str, float], constants) -> dict[str, float]:
    """Compute VBench quality/semantic/total from raw dimension scores."""
    normalized = {}
    for dim, val in raw_scores.items():
        key = dim.replace("_", " ")
        normalized[key] = normalize_score(key, val, constants[1])

    quality_dims = [
        "subject consistency", "background consistency",
        "temporal flickering", "motion smoothness",
        "dynamic degree", "aesthetic quality", "imaging quality",
    ]
    semantic_dims = ["temporal style", "overall consistency"]

    q_scores = [normalized[d] for d in quality_dims if d in normalized]
    q_weights = [constants[0][d] for d in quality_dims if d in normalized]
    quality = sum(s * w for s, w in zip(q_scores, q_weights)) / sum(q_weights) if q_weights else 0

    s_scores = [normalized[d] for d in semantic_dims if d in normalized]
    s_weights = [constants[0][d] for d in semantic_dims if d in normalized]
    semantic = sum(s * w for s, w in zip(s_scores, s_weights)) / sum(s_weights) if s_weights else 0

    total = (quality * constants[2] + semantic * constants[3]) / (constants[2] + constants[3])

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


def markdown_report(scores: dict[str, float], overall: dict[str, float], summary: dict) -> str:
    columns = [SHORT_NAMES[dim] for dim in DIMS] + ["quality", "semantic", "total"]
    values = [scores.get(dim) for dim in DIMS] + [
        overall["quality_score"], overall["semantic_score"], overall["total_score"],
    ]
    row = [f"{value:.4f}" if value is not None else "N/A" for value in values]
    return "\n".join([
        "# MXAttention Full — VBench Common-10 Results",
        "",
        f"- Videos: `{summary['video_dir']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Evaluation GPU: `{summary['cuda_visible_devices']}`",
        f"- VBench: `{summary['vbench_dir']}`",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" for _ in columns) + "|",
        "| " + " | ".join(row) + " |",
        "",
        "BF16 and NVFP4 deltas are intentionally deferred: matching artifacts are not available in this workspace.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description="All-in-one VBench evaluation")
    parser.add_argument("video_dir", help="Path to folder containing .mp4 videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cuda", default="2,3", help="CUDA_VISIBLE_DEVICES (default: 2,3)")
    parser.add_argument("--prompt-file", help="JSON video-name-to-prompt mapping for semantic dimensions")
    parser.add_argument("--vbench-dir", help="VBench checkout (default: $VBENCH_DIR)")
    parser.add_argument("--vbench-python", help="Python with VBench dependencies (default: $VBENCH_PYTHON)")
    parser.add_argument("--baseline", help="Path to baseline summary JSON for delta comparison")
    parser.add_argument("--output-dir", help="Override output directory (default: final-score/<basename>)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, just load existing results")
    args = parser.parse_args()
    vbench_dir, python_bin = vbench_paths(args)
    constants = vbench_constants(vbench_dir)

    video_dir = os.path.abspath(args.video_dir)
    basename = os.path.basename(video_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(__file__).parent / "final-score" / f"{basename}_custom_input_supported")

    os.makedirs(output_dir, exist_ok=True)

    # Run eval
    if not args.skip_eval:
        run_eval(video_dir, output_dir, args.seed, args.cuda, vbench_dir, python_bin, args.prompt_file)

    # Load scores
    scores = load_scores(output_dir)
    overall = compute_overall(scores, constants)

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
            baseline_overall = compute_overall(baseline_scores, constants)

    # Print results
    print_table(scores, overall, baseline_scores, baseline_overall)

    # Save JSON summary
    summary = {
        "video_dir": video_dir,
        "seed": args.seed,
        "num_videos": len(list(Path(video_dir).glob("*.mp4"))),
        "vbench_dir": str(vbench_dir),
        "cuda_visible_devices": args.cuda,
        "scores": scores,
        "overall": overall,
    }
    summary_file = Path(output_dir) / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    report_file = Path(output_dir) / "report.md"
    report_file.write_text(markdown_report(scores, overall, summary))
    print(f"Summary saved to: {summary_file}")
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
