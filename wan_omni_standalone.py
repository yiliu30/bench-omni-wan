#!/usr/bin/env python3
"""Standalone Wan2.2 text-to-video generation with vLLM-Omni."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "/dev/shm/.tmp_yi/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers/"
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight "
    "intensely on a spotlighted stage."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wan2.2 T2V offline through vLLM-Omni and save an MP4."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="t2v_out.mp4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    return parser.parse_args()


def build_text_to_video_prompt(prompt: str, negative_prompt: str | None) -> dict[str, Any]:
    request: dict[str, Any] = {
        "prompt": prompt,
        "modalities": ["video"],
    }
    if negative_prompt is not None:
        request["negative_prompt"] = negative_prompt
    return request


def extract_video_frames(result: Any) -> Any:
    from vllm_omni.outputs import OmniRequestOutput

    if isinstance(result, list):
        result = result[0] if result else None

    if isinstance(result, OmniRequestOutput):
        if not result.images:
            raise ValueError("No video frames found in OmniRequestOutput.")
        result = result.images[0]

    if isinstance(result, tuple) and len(result) == 2:
        result = result[0]
    if isinstance(result, dict):
        result = result.get("frames") or result.get("video")
    if isinstance(result, list) and len(result) == 1:
        inner = result[0]
        if isinstance(inner, (list, tuple, dict)):
            return extract_video_frames(inner)

    if result is None:
        raise ValueError("No video frames found in generation output.")
    return result


def normalize_for_export(frames: Any) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    def normalize_array(array: np.ndarray) -> np.ndarray:
        if array.ndim == 5:
            array = array[0]
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / 255.0
        return np.clip(array, 0.0, 1.0)

    def normalize_frame(frame: Any) -> Any:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.dim() == 3 and tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 0)
            return tensor.float().clamp(0, 1).numpy()
        if isinstance(frame, np.ndarray):
            return normalize_array(frame)
        if isinstance(frame, Image.Image):
            return np.asarray(frame).astype(np.float32) / 255.0
        return frame

    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            if video.shape[1] in (3, 4):
                video = video[0].permute(1, 2, 3, 0)
            else:
                video = video[0]
        elif video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        return video.float().clamp(0, 1).numpy()

    if isinstance(frames, np.ndarray):
        return normalize_array(frames)

    if isinstance(frames, list):
        if not frames:
            raise ValueError("No video frames found in generation output.")
        if len(frames) == 1 and isinstance(frames[0], np.ndarray) and frames[0].ndim in (4, 5):
            return normalize_array(frames[0])
        return [normalize_frame(frame) for frame in frames]

    return frames


def main() -> None:
    args = parse_args()
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import torch
    from diffusers.utils import export_to_video
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
    )
    pipe = Omni(
        model=args.model,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
    )

    generator = torch.Generator(
        device=current_omni_platform.device_type,
    ).manual_seed(args.seed)
    request = build_text_to_video_prompt(args.prompt, args.negative_prompt)
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Running Wan2.2 T2V with vLLM-Omni")
    print(f"  model: {args.model}")
    print(f"  output: {output_path}")
    print(f"  size: {args.width}x{args.height}, frames: {args.num_frames}, fps: {args.fps}")
    print(f"  steps: {args.num_inference_steps}, guidance: {args.guidance_scale}/{args.guidance_scale_2}")

    start = time.perf_counter()
    output = pipe.generate(request, sampling_params)
    frames = normalize_for_export(extract_video_frames(output))
    export_to_video(frames, str(output_path), fps=args.fps)
    elapsed = time.perf_counter() - start

    print(f"Saved generated video to {output_path} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
