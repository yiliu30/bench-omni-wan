#!/usr/bin/env python3
"""Extract scores for the common 10 videos across all configs and produce a comparison table.

Uses VBench's official scoring functions (scripts/cal_final_score.py) for normalization
and final score computation.
"""

import json
import sys
from pathlib import Path

# Reuse VBench's scoring logic
sys.path.insert(0, "/home/yiliu7/workspace/VBench/scripts")
from constant import (
    DIM_WEIGHT,
    NORMALIZE_DIC,
    QUALITY_LIST,
    QUALITY_WEIGHT,
    SEMANTIC_LIST,
    SEMANTIC_WEIGHT,
)
from cal_final_score import get_nomalized_score, get_quality_score, get_semantic_score, get_final_score

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

# The 10 videos present in bf16 baseline (and common to all)
COMMON_VIDEOS = [
    "a bicycle gliding through a snowy field-0.mp4",
    "a bicycle leaning against a tree-0.mp4",
    "a bicycle slowing down to stop-0.mp4",
    "a person drinking coffee in a cafe-0.mp4",
    "a person eating a burger-0.mp4",
    "a person giving a presentation to a room full of colleagues-0.mp4",
    "a person playing guitar-0.mp4",
    "a person swimming in ocean-0.mp4",
    "a person walking in the snowstorm-0.mp4",
    "a person washing the dishes-0.mp4",
]

CONFIGS = [
    "default_bf16_baseline_b200",
    "default_mxfp4_hw_attn_only",
    "default_mxfp4_linear_only_b200",
    "default_mxfp4_hw_attn_mxfp4_linear",
    "default_mxfp4_hw_rotation_attn_only",
    "default_mxfp8_hw_attn_only",
    "default_mxfp8_linear_only",
    "default_mxfp8_hw_attn_mxfp8_linear",
    "default_mxfp8_hw_attn_only_p_max_issue",

]


def find_result_file(config: str) -> Path | None:
    out_dir = RESULTS_DIR / f"{config}_custom_input_supported"
    if not out_dir.exists():
        return None
    results = sorted(out_dir.glob("*_eval_results.json"))
    return results[-1] if results else None


def extract_common_scores(result_file: Path) -> dict[str, float]:
    """Extract mean scores for only the common 10 videos."""
    data = json.loads(result_file.read_text())
    scores = {}
    for dim in DIMS:
        if dim not in data:
            continue
        dim_data = data[dim]
        per_video = dim_data[1]

        # Filter to common videos
        common_scores = []
        for entry in per_video:
            video_name = Path(entry["video_path"]).name
            if video_name in COMMON_VIDEOS:
                common_scores.append(entry["video_results"])

        if common_scores:
            mean = sum(common_scores) / len(common_scores)
            # imaging_quality per-video scores are raw (0-100), normalize to 0-1
            if dim == "imaging_quality" and mean > 1.0:
                mean = mean / 100.0
            scores[dim] = mean

    return scores


def compute_vbench_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Compute VBench quality/semantic/total scores using official normalization.

    raw_scores keys use underscores (e.g. "subject_consistency").
    VBench constants use spaces (e.g. "subject consistency").
    """
    # Convert to VBench key format (underscores -> spaces)
    upload_data = {}
    for dim, val in raw_scores.items():
        key = dim.replace("_", " ")
        upload_data[key] = val

    # Fill missing dims with 0 (required by VBench's normalization)
    from constant import TASK_INFO
    for key in TASK_INFO:
        if key not in upload_data:
            upload_data[key] = 0

    normalized = get_nomalized_score(upload_data)
    quality = get_quality_score(normalized)
    semantic = get_semantic_score(normalized)
    total = get_final_score(quality, semantic)

    # Compute quality score from dims we actually evaluated
    evaluated_quality_dims = [
        "subject consistency",
        "background consistency",
        "temporal flickering",
        "motion smoothness",
        "aesthetic quality",
        "imaging quality",
        "dynamic degree",
    ]
    eval_quality_scores = [normalized[d] for d in evaluated_quality_dims]
    eval_quality_weights = [DIM_WEIGHT[d] for d in evaluated_quality_dims]
    quality_partial = sum(eval_quality_scores) / sum(eval_quality_weights)

    # Semantic dims we evaluated
    evaluated_semantic_dims = [
        "temporal style",
        "overall consistency",
    ]
    eval_semantic_scores = [normalized[d] for d in evaluated_semantic_dims]
    eval_semantic_weights = [DIM_WEIGHT[d] for d in evaluated_semantic_dims]
    semantic_partial = sum(eval_semantic_scores) / sum(eval_semantic_weights)

    # Combined partial score using VBench weighting
    total_partial = (quality_partial * QUALITY_WEIGHT + semantic_partial * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

    return {
        "quality_score": quality_partial,
        "semantic_score": semantic_partial,
        "total_score": total_partial,
    }


def main():
    all_scores: dict[str, dict[str, float]] = {}
    all_vbench: dict[str, dict[str, float]] = {}

    for config in CONFIGS:
        result_file = find_result_file(config)
        if result_file is None:
            print(f"WARNING: No results for {config}")
            continue
        all_scores[config] = extract_common_scores(result_file)
        all_vbench[config] = compute_vbench_scores(all_scores[config])

    baseline = "default_bf16_baseline_b200"
    if baseline not in all_scores:
        print("ERROR: baseline not found")
        return

    # Short dimension names for table
    short_names = {
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

    # Print markdown table
    print("## VBench Comparison (Common 10 Videos, seed=42)\n")
    print("### Per-Dimension Scores\n")
    header = "| Config |"
    sep = "|--------|"
    for dim in DIMS:
        header += f" {short_names[dim]} |"
        sep += "-------:|"
    # Add final scores columns
    header += " quality | semantic | **total** |"
    sep += "-------:|-------:|-------:|"
    print(header)
    print(sep)

    # Score rows
    for config in CONFIGS:
        scores = all_scores[config]
        vb = all_vbench[config]
        row = f"| {config} |"
        for dim in DIMS:
            val = scores.get(dim)
            row += f" {val:.4f} |" if val is not None else " N/A |"
        row += f" {vb['quality_score']:.4f} | {vb['semantic_score']:.4f} | **{vb['total_score']:.4f}** |"
        print(row)

    # Delta rows (in %)
    print("")
    print("### Delta vs BF16 Baseline (%)\n")
    header2 = "| Config |"
    sep2 = "|--------|"
    for dim in DIMS:
        header2 += f" {short_names[dim]} |"
        sep2 += "-------:|"
    header2 += " quality | semantic | **total** |"
    sep2 += "-------:|-------:|-------:|"
    print(header2)
    print(sep2)

    baseline_scores = all_scores[baseline]
    baseline_vb = all_vbench[baseline]
    for config in CONFIGS:
        if config == baseline:
            continue
        scores = all_scores[config]
        vb = all_vbench[config]
        row = f"| {config.replace('default_', '')} |"
        for dim in DIMS:
            base_val = baseline_scores.get(dim)
            val = scores.get(dim)
            if val is not None and base_val is not None:
                pct = (val - base_val) / base_val * 100
                sign = "+" if pct >= 0 else ""
                row += f" {sign}{pct:.2f}% |"
            else:
                row += " N/A |"
        # Final score deltas
        for key in ["quality_score", "semantic_score", "total_score"]:
            pct = (vb[key] - baseline_vb[key]) / baseline_vb[key] * 100
            sign = "+" if pct >= 0 else ""
            bold = "**" if key == "total_score" else ""
            row += f" {bold}{sign}{pct:.2f}%{bold} |"
        print(row)

    # Save JSON
    output = {
        "common_videos": COMMON_VIDEOS,
        "scores": all_scores,
        "vbench_scores": all_vbench,
        "deltas_pct": {},
    }
    for config in CONFIGS:
        if config == baseline:
            continue
        deltas = {}
        for dim in DIMS:
            base_val = baseline_scores.get(dim)
            val = all_scores[config].get(dim)
            if val is not None and base_val is not None:
                deltas[dim] = (val - base_val) / base_val * 100
        vb = all_vbench[config]
        for key in ["quality_score", "semantic_score", "total_score"]:
            deltas[key] = (vb[key] - baseline_vb[key]) / baseline_vb[key] * 100
        output["deltas_pct"][config] = deltas

    out_file = RESULTS_DIR / "summary_common10.json"
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nJSON saved to: {out_file}")

    # Save markdown doc
    md_file = Path(__file__).parent / "docs" / "vbench_common10_results.md"
    md_file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# VBench Evaluation Results — Common 10 Videos\n")
    lines.append("**Evaluation settings:** seed=42, PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN\n")
    lines.append("**Videos evaluated:** 10 prompts common across all configs (person/bicycle scenes)\n")
    lines.append("**Scoring:** VBench official normalization + weighted aggregation")
    lines.append("  - Total = (Quality × 4 + Semantic × 1) / 5")
    lines.append("  - Quality dims: subject_con, bg_con, temp_flk, mot_smo, aesth_q, img_q")
    lines.append("  - Semantic dims: overall_con, temp_style\n")
    lines.append("## Per-Dimension Scores\n")

    header = "| Config |"
    sep = "|--------|"
    for dim in DIMS:
        header += f" {short_names[dim]} |"
        sep += "-------:|"
    header += " quality | semantic | **total** |"
    sep += "-------:|-------:|-------:|"
    lines.append(header)
    lines.append(sep)
    for config in CONFIGS:
        scores = all_scores[config]
        vb = all_vbench[config]
        row = f"| {config} |"
        for dim in DIMS:
            val = scores.get(dim)
            row += f" {val:.4f} |" if val is not None else " N/A |"
        row += f" {vb['quality_score']:.4f} | {vb['semantic_score']:.4f} | **{vb['total_score']:.4f}** |"
        lines.append(row)

    lines.append("")
    lines.append("## Delta vs BF16 Baseline (%)\n")
    lines.append(header.replace("Config", "Config "))
    lines.append(sep)
    for config in CONFIGS:
        if config == baseline:
            continue
        scores = all_scores[config]
        vb = all_vbench[config]
        row = f"| {config.replace('default_', '')} |"
        for dim in DIMS:
            base_val = baseline_scores.get(dim)
            val = scores.get(dim)
            if val is not None and base_val is not None:
                pct = (val - base_val) / base_val * 100
                sign = "+" if pct >= 0 else ""
                row += f" {sign}{pct:.2f}% |"
            else:
                row += " N/A |"
        for key in ["quality_score", "semantic_score", "total_score"]:
            pct = (vb[key] - baseline_vb[key]) / baseline_vb[key] * 100
            sign = "+" if pct >= 0 else ""
            bold = "**" if key == "total_score" else ""
            row += f" {bold}{sign}{pct:.2f}%{bold} |"
        lines.append(row)

    lines.append("")
    lines.append("## Dimension Descriptions\n")
    lines.append("- **subj_con**: Subject consistency (DINO feature similarity across frames)")
    lines.append("- **bg_con**: Background consistency (CLIP feature similarity)")
    lines.append("- **temp_flk**: Temporal flickering (lower flickering = higher score)")
    lines.append("- **mot_smo**: Motion smoothness (AMT-based flow estimation)")
    lines.append("- **aesth_q**: Aesthetic quality (LAION aesthetic predictor)")
    lines.append("- **img_q**: Imaging quality (MUSIQ no-reference IQA, normalized 0-1)")
    lines.append("- **overall_con**: Overall consistency (ViCLIP text-video similarity)")
    lines.append("- **temp_style**: Temporal style (ViCLIP temporal coherence)")
    lines.append("- **quality**: Weighted avg of quality dims (VBench normalized)")
    lines.append("- **semantic**: Weighted avg of semantic dims (VBench normalized)")
    lines.append("- **total**: (quality × 4 + semantic × 1) / 5")
    lines.append("")

    md_file.write_text("\n".join(lines))
    print(f"Markdown saved to: {md_file}")


if __name__ == "__main__":
    main()
