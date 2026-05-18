# VBench Evaluation Dimensions Reference

## Quick Reference Table

| Dimension | Category | Needs Prompt? | Model Used | Custom Input? | What It Measures |
|---|---|---|---|---|---|
| `subject_consistency` | Quality | No | DINO ViT-B/16 | ✅ | Subject identity stays consistent across frames |
| `background_consistency` | Quality | No | CLIP ViT-B/32 | ✅ | Background doesn't change unexpectedly |
| `temporal_flickering` | Quality | No | **None** (pixel MAE) | ✅ | Frame-to-frame pixel-level stability |
| `motion_smoothness` | Quality | No | AMT-S (frame interpolation) | ✅ | Motion looks natural, no jerky transitions |
| `dynamic_degree` | Quality | No | RAFT (optical flow) | ✅ | How much motion exists (penalizes static videos) |
| `aesthetic_quality` | Quality | No | CLIP ViT-L/14 + LAION aesthetic head | ✅ | Visual beauty/appeal per frame |
| `imaging_quality` | Quality | No | MUSIQ-SPAQ (IQA model) | ✅ | Technical image quality (sharpness, noise, etc.) |
| `overall_consistency` | Semantic | **Yes** | ViCLIP | ❌ | Text-video alignment via video-text similarity |
| `temporal_style` | Semantic | **Yes** | ViCLIP | ❌ | Temporal style matches the prompt |
| `appearance_style` | Semantic | **Yes** | CLIP ViT-B/32 | ❌ | Visual style matches the prompt |
| `object_class` | Semantic | **Yes** | GRiT (dense captioning) + detectron2 | ❌ | Correct objects appear in the video |
| `multiple_objects` | Semantic | **Yes** | GRiT + detectron2 | ❌ | Correct count of objects |
| `color` | Semantic | **Yes** | GRiT + detectron2 | ❌ | Object colors match the prompt |
| `spatial_relationship` | Semantic | **Yes** | GRiT + detectron2 | ❌ | Object spatial layout matches prompt |
| `scene` | Semantic | **Yes** | Tag2Text (Swin-B) | ❌ | Scene type matches the prompt |
| `human_action` | Semantic | **Yes** | UMT (ViT-L, action recognition) | ✅* | Action in video matches the prompt |

- **Custom Input = ✅**: can use `--mode=custom_input` with your own videos.
- **Custom Input = ❌**: requires VBench's `VBench_full_info.json` metadata (structured prompts with auxiliary info).
- `human_action` (✅*): supports custom input but still needs a prompt to compare against.

## How Each Dimension Works

### Quality Dimensions (no prompt needed)

#### `subject_consistency` — DINO ViT-B/16
Extracts DINO features per frame. For each frame i (from frame 2 onwards):
- `sim_pre` = cosine similarity with previous frame (temporal coherence)
- `sim_fir` = cosine similarity with first frame (identity preservation)
- Frame score = `(sim_pre + sim_fir) / 2`

Per-video score = average of all frame scores. Overall = average across videos.

#### `background_consistency` — CLIP ViT-B/32
Same approach as subject_consistency but uses CLIP features instead of DINO. CLIP captures more semantic/scene-level information, making it better suited for background evaluation.

#### `temporal_flickering` — No model (pixel MAE)
Computes mean absolute error (MAE) between consecutive frames in pixel space.
- Score = `(255 - mean_MAE) / 255`
- Higher score = less flickering (more stable)
- This is the only dimension that requires no pretrained model at all.

#### `motion_smoothness` — AMT-S (frame interpolation)
Uses AMT-S (Adaptive Multi-scale Temporal model) to interpolate between frames, then measures how well the interpolated frame matches the actual frame. Smooth motion = easy to interpolate = high score.

#### `dynamic_degree` — RAFT (optical flow)
Uses RAFT optical flow to measure the magnitude of motion in the video. This penalizes completely static videos — a video generation model should produce videos with actual movement.

#### `aesthetic_quality` — CLIP ViT-L/14 + LAION aesthetic linear head
Extracts CLIP ViT-L/14 features per frame, passes them through a linear aesthetic predictor trained by LAION. Scores each frame for visual appeal/beauty, then averages.

#### `imaging_quality` — MUSIQ-SPAQ
Uses MUSIQ (Multi-Scale Image Quality Transformer) trained on SPAQ dataset. Evaluates technical quality: sharpness, noise, compression artifacts, etc. per frame.

### Semantic Dimensions (prompt required)

#### `overall_consistency` — ViCLIP
Uses ViCLIP (Video-CLIP) to compute video-level embeddings and text embeddings, then measures cosine similarity. Checks if the generated video matches the text prompt semantically.

#### `temporal_style` — ViCLIP
Similar to overall_consistency but focused on temporal/motion style (e.g., "time-lapse", "slow-motion").

#### `appearance_style` — CLIP ViT-B/32
Uses CLIP to check if visual appearance style matches the prompt (e.g., "watercolor", "cyberpunk").

#### `object_class` — GRiT + detectron2
Uses GRiT (dense captioning model) to detect objects in video frames, then checks if the prompted object class actually appears.

#### `multiple_objects` — GRiT + detectron2
Same detection pipeline as object_class, but checks if the correct number of objects are present.

#### `color` — GRiT + detectron2
Detects objects and their colors, checks against the prompted color.

#### `spatial_relationship` — GRiT + detectron2
Detects objects and their spatial positions, checks if the spatial layout matches the prompt (e.g., "a cat to the left of a dog").

#### `scene` — Tag2Text (Swin-B)
Uses Tag2Text model to recognize the scene type in video frames, then matches against the prompted scene.

#### `human_action` — UMT (ViT-L)
Uses UMT (Unified Multimodal Transformer) for action recognition. Classifies the action in the video and checks if it matches the prompted action.

## Scoring

### Normalization (for leaderboard)
Each dimension score is normalized: `(score - min) / (max - min)` using constants from `VBench/scripts/constant.py`.

### Final Score Composition
- **Quality Score** = weighted average of 7 quality dimensions (dynamic_degree has weight 0.5, others 1.0)
- **Semantic Score** = weighted average of 9 semantic dimensions (all weight 1.0)
- **Total Score** = `4 × Quality Score + 1 × Semantic Score` (quality-weighted 4:1)

## Running Evaluation

### VBench env
```bash
# Env location
/home/yiliu7/workspace/venvs/vbench

# Python binary
/home/yiliu7/workspace/venvs/vbench/bin/python
```

### Evaluate subject_consistency (tested example)
```bash
cd /home/yiliu7/workspace/VBench

/home/yiliu7/workspace/venvs/vbench/bin/python evaluate.py \
    --videos_path /home/yiliu7/workspace/vllm-omni/bench-wan/output/default_bf16 \
    --dimension subject_consistency \
    --mode=custom_input
```

Result from 2026-05-18 run: **0.9639** across 72 videos.
Output: `./evaluation_results/results_2026-05-18-01:20:55_eval_results.json`

### Evaluate all quality dimensions at once
```bash
cd /home/yiliu7/workspace/VBench

/home/yiliu7/workspace/venvs/vbench/bin/python evaluate.py \
    --videos_path /home/yiliu7/workspace/vllm-omni/bench-wan/output/default_bf16 \
    --dimension subject_consistency background_consistency temporal_flickering \
               motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    --mode=custom_input
```

### Generic form (for any video directory)
```bash
cd /home/yiliu7/workspace/VBench

/home/yiliu7/workspace/venvs/vbench/bin/python evaluate.py \
    --videos_path /path/to/videos \
    --dimension <dim1> [<dim2> ...] \
    --mode=custom_input
```

### Multi-GPU
```bash
cd /home/yiliu7/workspace/VBench

torchrun --nproc_per_node=4 --standalone evaluate.py \
    --videos_path /path/to/videos \
    --dimension subject_consistency \
    --mode=custom_input
```

### Results
Saved to `./evaluation_results/` as JSON. Format:
```json
{
    "dimension_name": [
        0.9639,          // overall score
        [                // per-video scores
            {"video_path": "...", "video_results": 0.99},
            ...
        ]
    ]
}
```

## Video Naming Convention
VBench expects videos named `{prompt}-{index}.mp4` (e.g., `a cat walking-0.mp4`). The prompt is extracted from the filename by stripping the `-N` suffix. For standard VBench evaluation, 5 videos per prompt (`-0` through `-4`) are expected.

## Model Cache
All pretrained models auto-download to `~/.cache/vbench/`. Override with `VBENCH_CACHE_DIR` env var.
