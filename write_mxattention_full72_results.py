#!/usr/bin/env python3
"""Append the completed full-72 VBench result to the shared results document."""
import argparse
import json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--results-doc", type=Path, default=Path("docs/vbench_common10_results.md"))
    a = p.parse_args()
    summary = json.loads(a.summary.read_text())
    scores, overall = summary["scores"], summary["overall"]
    order = ["subject_consistency", "background_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality", "overall_consistency", "temporal_style"]
    values = [scores[k] for k in order] + [overall["quality_score"], overall["semantic_score"], overall["total_score"]]
    row = "| mxattention_full_72 | " + " | ".join(f"{v:.4f}" for v in values[:-1]) + f" | **{values[-1]:.4f}** |"
    heading = "## MXAttention Full-72 Results"
    text = a.results_doc.read_text()
    if heading in text:
        start = text.index(heading)
        text = text[:start].rstrip() + "\n\n"
    section = "\n".join([heading, "", "- 72 videos; seed=42; TP=1; MXAttention full (qmax 7.25, Hadamard, SAGE_ATTN).", "- Provenance: 19 resumed clips have fresh worker manifests; 53 preserved clips use documented configuration evidence in `full72/legacy-evidence.json`.", "- Quality uses all seven VBench quality dimensions, including dynamic degree.", "", "| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", row, ""])
    a.results_doc.write_text(text + section)
    print(f"Updated {a.results_doc}")

if __name__ == "__main__":
    main()
