# VBench Evaluation Results — Common 10 Videos

**Evaluation settings:** seed=42, PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN

**Videos evaluated:** 10 prompts common across all configs (person/bicycle scenes)

**Scoring:** VBench official normalization + weighted aggregation
  - Total = (Quality × 4 + Semantic × 1) / 5
  - Quality dims: subject_con, bg_con, temp_flk, mot_smo, aesth_q, img_q
  - Semantic dims: overall_con, temp_style

## Per-Dimension Scores

| Config | subj_con | bg_con | temp_flk | mot_smo | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| default_bf16_baseline_b200 | 0.9408 | 0.9434 | 0.9704 | 0.9843 | 0.5825 | 0.6964 | 0.2249 | 0.2249 | 0.8347 | 0.6180 | **0.7913** |
| default_mxfp4_hw_attn_only | 0.9358 | 0.9509 | 0.9759 | 0.9866 | 0.5643 | 0.6203 | 0.2202 | 0.2202 | 0.8235 | 0.6049 | **0.7798** |
| default_mxfp4_hw_rotation_attn_only | 0.9354 | 0.9485 | 0.9729 | 0.9852 | 0.5853 | 0.6423 | 0.2211 | 0.2211 | 0.8278 | 0.6075 | **0.7837** |
| default_mxfp8_hw_attn_only | 0.9359 | 0.9391 | 0.9698 | 0.9840 | 0.5722 | 0.6818 | 0.2210 | 0.2210 | 0.8281 | 0.6070 | **0.7839** |
| default_mxfp8_linear_only | 0.9273 | 0.9456 | 0.9694 | 0.9839 | 0.5896 | 0.6875 | 0.2206 | 0.2206 | 0.8315 | 0.6061 | **0.7864** |

## Delta vs BF16 Baseline (%)

| Config  | subj_con | bg_con | temp_flk | mot_smo | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| mxfp4_hw_attn_only | -0.53% | +0.79% | +0.57% | +0.23% | -3.12% | -10.93% | -2.11% | -2.11% | -1.34% | -2.11% | **-1.46%** |
| mxfp4_hw_rotation_attn_only | -0.58% | +0.53% | +0.26% | +0.08% | +0.49% | -7.76% | -1.69% | -1.69% | -0.82% | -1.69% | **-0.96%** |
| mxfp8_hw_attn_only | -0.52% | -0.46% | -0.06% | -0.03% | -1.77% | -2.10% | -1.77% | -1.77% | -0.78% | -1.77% | **-0.94%** |
| mxfp8_linear_only | -1.44% | +0.22% | -0.10% | -0.04% | +1.22% | -1.28% | -1.93% | -1.93% | -0.38% | -1.93% | **-0.62%** |

## Dimension Descriptions

- **subj_con**: Subject consistency (DINO feature similarity across frames)
- **bg_con**: Background consistency (CLIP feature similarity)
- **temp_flk**: Temporal flickering (lower flickering = higher score)
- **mot_smo**: Motion smoothness (AMT-based flow estimation)
- **aesth_q**: Aesthetic quality (LAION aesthetic predictor)
- **img_q**: Imaging quality (MUSIQ no-reference IQA, normalized 0-1)
- **overall_con**: Overall consistency (ViCLIP text-video similarity)
- **temp_style**: Temporal style (ViCLIP temporal coherence)
- **quality**: Weighted avg of quality dims (VBench normalized)
- **semantic**: Weighted avg of semantic dims (VBench normalized)
- **total**: (quality × 4 + semantic × 1) / 5
