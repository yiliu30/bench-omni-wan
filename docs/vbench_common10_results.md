# VBench Evaluation Results — Common 10 Videos

**Evaluation settings:** seed=42, PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN

**Videos evaluated:** 10 prompts common across all configs (person/bicycle scenes)

**Scoring:** VBench official normalization + weighted aggregation
  - Total = (Quality × 4 + Semantic × 1) / 5
  - Quality dims: subject_con, bg_con, temp_flk, mot_smo, aesth_q, img_q
  - Semantic dims: overall_con, temp_style

## Per-Dimension Scores

| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| default_bf16_baseline_b200 | 0.9408 | 0.9434 | 0.9704 | 0.9843 | 0.7000 | 0.5825 | 0.6964 | 0.2249 | 0.2249 | 0.8243 | 0.6180 | **0.7830** |
| default_nvfp4_hw_attn_only | 0.9388 | 0.9448 | 0.9694 | 0.9819 | 0.7000 | 0.5789 | 0.6508 | 0.2209 | 0.2209 | 0.8150 | 0.6068 | **0.7733** |
| default_mxfp4_hw_rotation_attn_only | 0.9354 | 0.9485 | 0.9729 | 0.9852 | 0.5000 | 0.5853 | 0.6423 | 0.2211 | 0.2211 | 0.8026 | 0.6075 | **0.7636** |
| default_mxfp4_hw_attn_only | 0.9358 | 0.9509 | 0.9759 | 0.9866 | 0.5000 | 0.5643 | 0.6203 | 0.2202 | 0.2202 | 0.7986 | 0.6049 | **0.7598** |
| default_mxfp8_hw_attn_only | 0.9325 | 0.9421 | 0.9716 | 0.9847 | 0.6000 | 0.5778 | 0.6790 | 0.2200 | 0.2200 | 0.8121 | 0.6043 | **0.7706** |
| default_mxfp8_linear_only | 0.9273 | 0.9456 | 0.9694 | 0.9839 | 0.7000 | 0.5896 | 0.6875 | 0.2206 | 0.2206 | 0.8214 | 0.6061 | **0.7783** |
| default_mxfp8_hw_attn_mxfp8_linear | 0.9313 | 0.9375 | 0.9717 | 0.9853 | 0.7000 | 0.5923 | 0.6883 | 0.2256 | 0.2256 | 0.8227 | 0.6197 | **0.7821** |
| mxattention_full_common10 | 0.9351 | 0.9448 | 0.9726 | 0.9843 | 0.7000 | 0.5558 | 0.6523 | 0.2262 | 0.2262 | 0.8230 | 0.6213 | **0.7827** |

## Delta vs BF16 Baseline (%)

| Config  | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| nvfp4_hw_attn_only | -0.21% | +0.15% | -0.10% | -0.25% | +0.00% | -0.62% | -6.55% | -1.80% | -1.80% | -1.13% | -1.80% | **-1.24%** |
| mxfp4_hw_rotation_attn_only | -0.58% | +0.53% | +0.26% | +0.08% | -28.57% | +0.49% | -7.76% | -1.69% | -1.69% | -2.64% | -1.69% | **-2.49%** |
| mxfp4_hw_attn_only | -0.53% | +0.79% | +0.57% | +0.23% | -28.57% | -3.12% | -10.93% | -2.11% | -2.11% | -3.12% | -2.11% | **-2.96%** |
| mxfp8_hw_attn_only | -0.88% | -0.14% | +0.12% | +0.04% | -14.29% | -0.80% | -2.49% | -2.21% | -2.21% | -1.48% | -2.21% | **-1.59%** |
| mxfp8_linear_only | -1.44% | +0.22% | -0.10% | -0.04% | +0.00% | +1.22% | -1.28% | -1.93% | -1.93% | -0.35% | -1.93% | **-0.60%** |
| mxfp8_hw_attn_mxfp8_linear | -1.01% | -0.63% | +0.13% | +0.09% | +0.00% | +1.68% | -1.16% | +0.29% | +0.29% | -0.20% | +0.29% | **-0.12%** |
| mxattention_full_common10 | -0.61% | +0.15% | +0.22% | +0.00% | +0.00% | -4.59% | -6.33% | +0.56% | +0.56% | -0.16% | +0.53% | **-0.04%** |

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

## MXAttention Full-72 Results

- 72 videos; seed=42; TP=1; MXAttention full (qmax 7.25, Hadamard, SAGE_ATTN).
- Provenance: 19 resumed clips have fresh worker manifests; 53 preserved clips use documented configuration evidence in `full72/legacy-evidence.json`.
- Quality uses all seven VBench quality dimensions, including dynamic degree.

| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mxattention_full_72 | 0.9438 | 0.9517 | 0.9653 | 0.9795 | 0.6111 | 0.5573 | 0.6843 | 0.2183 | 0.2183 | 0.8093 | 0.5997 | **0.7674** |
