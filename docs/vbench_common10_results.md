# VBench Evaluation Results — Common 10 Videos

**Evaluation settings:** seed=42, PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN

**Videos evaluated:** 10 prompts common across all configs (person/bicycle scenes)

**Scoring:** VBench official normalization + weighted aggregation
  - Total = (Quality × 4 + Semantic × 1) / 5
  - Quality dims: subject_con, bg_con, temp_flk, mot_smo, aesth_q, img_q
  - Semantic dims: overall_con, temp_style

## Per-Dimension Scores

| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | human_act | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| default_bf16_baseline_b200 | 0.9408 | 0.9434 | 0.9704 | 0.9843 | 0.7000 | 0.5825 | 0.6964 | 0.2249 | 0.2249 | N/A | 0.8243 | 0.6180 | **0.7830** |
| default_mxfp4_hw_attn_only | 0.9358 | 0.9509 | 0.9759 | 0.9866 | 0.5000 | 0.5643 | 0.6203 | 0.2202 | 0.2202 | N/A | 0.7986 | 0.6049 | **0.7598** |
| default_mxfp4_linear_only_b200 | 0.9492 | 0.9531 | 0.9756 | 0.9862 | 0.6000 | 0.5866 | 0.6459 | 0.2184 | 0.2184 | N/A | 0.8161 | 0.5999 | **0.7729** |
| default_mxfp4_hw_attn_mxfp4_linear | 0.9441 | 0.9609 | 0.9805 | 0.9875 | 0.5000 | 0.5530 | 0.5975 | 0.2191 | 0.2191 | N/A | 0.7993 | 0.6019 | **0.7598** |
| default_mxfp4_hw_rotation_attn_only | 0.9354 | 0.9485 | 0.9729 | 0.9852 | 0.5000 | 0.5853 | 0.6423 | 0.2211 | 0.2211 | N/A | 0.8026 | 0.6075 | **0.7636** |
| default_mxfp8_hw_attn_only | 0.9268 | 0.9426 | 0.9727 | 0.9842 | 0.6000 | 0.5912 | 0.6850 | 0.2197 | 0.2197 | N/A | 0.8144 | 0.6037 | **0.7722** |
| default_mxfp8_linear_only | 0.9273 | 0.9456 | 0.9694 | 0.9839 | 0.7000 | 0.5896 | 0.6875 | 0.2206 | 0.2206 | N/A | 0.8214 | 0.6061 | **0.7783** |
| default_mxfp8_hw_attn_mxfp8_linear | 0.9380 | 0.9399 | 0.9741 | 0.9863 | 0.6000 | 0.5765 | 0.6787 | 0.2201 | 0.2201 | N/A | 0.8143 | 0.6046 | **0.7724** |
| default_mxfp8_hw_attn_only_p_max_issue | 0.9359 | 0.9391 | 0.9698 | 0.9840 | 0.7000 | 0.5722 | 0.6818 | 0.2210 | 0.2210 | N/A | 0.8183 | 0.6070 | **0.7760** |

## Delta vs BF16 Baseline (%)

| Config  | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | human_act | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| mxfp4_hw_attn_only | -0.53% | +0.79% | +0.57% | +0.23% | -28.57% | -3.12% | -10.93% | -2.11% | -2.11% | N/A | -3.12% | -2.11% | **-2.96%** |
| mxfp4_linear_only_b200 | +0.89% | +1.02% | +0.54% | +0.19% | -14.29% | +0.70% | -7.26% | -2.92% | -2.92% | N/A | -0.99% | -2.92% | **-1.30%** |
| mxfp4_hw_attn_mxfp4_linear | +0.35% | +1.85% | +1.04% | +0.32% | -28.57% | -5.07% | -14.21% | -2.60% | -2.60% | N/A | -3.04% | -2.60% | **-2.97%** |
| mxfp4_hw_rotation_attn_only | -0.58% | +0.53% | +0.26% | +0.08% | -28.57% | +0.49% | -7.76% | -1.69% | -1.69% | N/A | -2.64% | -1.69% | **-2.49%** |
| mxfp8_hw_attn_only | -1.49% | -0.09% | +0.24% | -0.01% | -14.29% | +1.49% | -1.64% | -2.31% | -2.31% | N/A | -1.21% | -2.31% | **-1.38%** |
| mxfp8_linear_only | -1.44% | +0.22% | -0.10% | -0.04% | +0.00% | +1.22% | -1.28% | -1.93% | -1.93% | N/A | -0.35% | -1.93% | **-0.60%** |
| mxfp8_hw_attn_mxfp8_linear | -0.31% | -0.37% | +0.38% | +0.20% | -14.29% | -1.03% | -2.54% | -2.17% | -2.17% | N/A | -1.21% | -2.17% | **-1.36%** |
| mxfp8_hw_attn_only_p_max_issue | -0.52% | -0.46% | -0.06% | -0.03% | +0.00% | -1.77% | -2.10% | -1.77% | -1.77% | N/A | -0.73% | -1.77% | **-0.90%** |

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
