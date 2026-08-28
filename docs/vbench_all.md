# VBench Evaluation Results — Full 72 Videos

**Evaluation settings:** seed=42, PYTHONHASHSEED=42, CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN

**Videos evaluated:** 72 prompts (full set; configs with only the common 10 are excluded)

**Configs included:** default_bf16_baseline_b200, default_mxfp8_hw_attn_only, default_mxfp8_linear_only, default_mxfp8_hw_attn_mxfp8_linear, default_mxfp8_hw_attn_mxfp8_linear_mixed_mxfp4_attn1, default_mxfp8_hw_attn_mxfp4_linear_mixed_mxfp4_attn1, default_mixed_mxfp4qk_mxfp8pv_hw_attn_mxfp8_linear.env, default_mixed_mxfp8qk_mxfp4pv_hw_attn_mxfp8_linear, default_mxfp4_hw_attn_only, default_mxfp4_hw_attn_mxfp4_linear

**Scoring:** VBench official normalization + weighted aggregation
  - Total = (Quality × 4 + Semantic × 1) / 5
  - Quality dims: subject_con, bg_con, temp_flk, mot_smo, aesth_q, img_q, dyn_deg
  - Semantic dims: overall_con, temp_style

## Per-Dimension Scores

| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| default_bf16_baseline_b200 | 0.9514 | 0.9583 | 0.9621 | 0.9792 | 0.6389 | 0.5909 | 0.6920 | 0.2201 | 0.2201 | 0.8191 | 0.6046 | **0.7762** |
| default_mxfp8_hw_attn_only | 0.9504 | 0.9567 | 0.9626 | 0.9793 | 0.6250 | 0.5917 | 0.6894 | 0.2194 | 0.2194 | 0.8175 | 0.6026 | **0.7745** |
| default_mxfp8_linear_only | 0.9491 | 0.9549 | 0.9630 | 0.9792 | 0.6667 | 0.5929 | 0.6929 | 0.2202 | 0.2202 | 0.8209 | 0.6050 | **0.7777** |
| default_mxfp8_hw_attn_mxfp8_linear | 0.9498 | 0.9555 | 0.9637 | 0.9798 | 0.6389 | 0.5924 | 0.6932 | 0.2201 | 0.2201 | 0.8196 | 0.6046 | **0.7766** |
| default_mxfp8_hw_attn_mxfp8_linear_mixed_mxfp4_attn1 | 0.9442 | 0.9490 | 0.9668 | 0.9812 | 0.6111 | 0.5763 | 0.6864 | 0.2210 | 0.2210 | 0.8136 | 0.6072 | **0.7723** |
| default_mxfp8_hw_attn_mxfp4_linear_mixed_mxfp4_attn1 | 0.9420 | 0.9543 | 0.9715 | 0.9829 | 0.6250 | 0.5698 | 0.6610 | 0.2236 | 0.2236 | 0.8133 | 0.6143 | **0.7735** |
| default_mixed_mxfp4qk_mxfp8pv_hw_attn_mxfp8_linear.env | 0.9359 | 0.9524 | 0.9778 | 0.9856 | 0.5417 | 0.5566 | 0.6686 | 0.2207 | 0.2207 | 0.8086 | 0.6062 | **0.7681** |
| default_mixed_mxfp8qk_mxfp4pv_hw_attn_mxfp8_linear | 0.9435 | 0.9538 | 0.9627 | 0.9793 | 0.6389 | 0.5856 | 0.6775 | 0.2209 | 0.2209 | 0.8140 | 0.6069 | **0.7725** |
| default_mxfp4_hw_attn_only | 0.9409 | 0.9526 | 0.9651 | 0.9792 | 0.6250 | 0.5753 | 0.6683 | 0.2229 | 0.2229 | 0.8102 | 0.6124 | **0.7706** |
| default_mxfp4_hw_attn_mxfp4_linear | 0.9408 | 0.9541 | 0.9716 | 0.9824 | 0.6111 | 0.5625 | 0.6615 | 0.2218 | 0.2218 | 0.8107 | 0.6094 | **0.7705** |
| mxattention_full_72 | 0.9438 | 0.9517 | 0.9653 | 0.9795 | 0.6111 | 0.5573 | 0.6843 | 0.2183 | 0.2183 | 0.8093 | 0.5997 | **0.7674** |

## Delta vs BF16 Baseline (%)

| Config | subj_con | bg_con | temp_flk | mot_smo | dyn_deg | aesth_q | img_q | overall_con | temp_style | quality | semantic | **total** |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| mxfp8_hw_attn_only | -0.11% | -0.17% | +0.05% | +0.02% | -2.17% | +0.13% | -0.39% | -0.33% | -0.33% | -0.20% | -0.33% | **-0.22%** |
| mxfp8_linear_only | -0.25% | -0.36% | +0.09% | +0.01% | +4.35% | +0.33% | +0.13% | +0.07% | +0.07% | +0.22% | +0.07% | **+0.20%** |
| mxfp8_hw_attn_mxfp8_linear | -0.17% | -0.30% | +0.16% | +0.07% | +0.00% | +0.25% | +0.17% | -0.00% | -0.00% | +0.06% | -0.00% | **+0.05%** |
| mxfp8_hw_attn_mxfp8_linear_mixed_mxfp4_attn1 | -0.75% | -0.98% | +0.49% | +0.21% | -4.35% | -2.47% | -0.82% | +0.42% | +0.42% | -0.67% | +0.42% | **-0.50%** |
| mxfp8_hw_attn_mxfp4_linear_mixed_mxfp4_attn1 | -0.99% | -0.42% | +0.97% | +0.38% | -2.17% | -3.58% | -4.49% | +1.60% | +1.60% | -0.71% | +1.60% | **-0.35%** |
| mixed_mxfp4qk_mxfp8pv_hw_attn_mxfp8_linear.env | -1.63% | -0.62% | +1.63% | +0.66% | -15.22% | -5.81% | -3.39% | +0.27% | +0.27% | -1.28% | +0.27% | **-1.04%** |
| mixed_mxfp8qk_mxfp4pv_hw_attn_mxfp8_linear | -0.83% | -0.48% | +0.06% | +0.02% | +0.00% | -0.91% | -2.10% | +0.37% | +0.37% | -0.63% | +0.37% | **-0.47%** |
| mxfp4_hw_attn_only | -1.10% | -0.60% | +0.31% | +0.01% | -2.17% | -2.64% | -3.43% | +1.29% | +1.29% | -1.09% | +1.29% | **-0.72%** |
| mxfp4_hw_attn_mxfp4_linear | -1.12% | -0.45% | +0.98% | +0.33% | -4.35% | -4.80% | -4.41% | +0.78% | +0.78% | -1.02% | +0.78% | **-0.74%** |
| mxattention_full_72 | -0.80% | -0.68% | +0.34% | +0.03% | -4.35% | -5.70% | -1.11% | -0.84% | -0.84% | -1.77% | -0.83% | **-1.13%** |

## MXAttention Full-72 Notes

- Evaluated 72 videos with seed 42 on GPU 1 using the nine supported custom-input dimensions.
- Configuration: MXAttention full, qmax 7.25, Hadamard enabled, and SAGE_ATTN.
- All artifacts passed 1280×720, 81-frame, 16-FPS validation. The 19 resumed clips have fresh worker manifests; the other 53 retain documented configuration evidence.
- Quality includes all seven VBench quality dimensions, including dynamic degree.
