"""Preload the local vLLM FlashAttention extension for Wan runs.

This avoids the torchvision NMS shim that was needed before torchvision was
updated in the Omni venv.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def _preload_local_vllm_flash_attn() -> None:
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    source_dir = "/dev/shm/.tmp_yi/workspace/vllm/vllm/vllm_flash_attn"
    spec = importlib.util.spec_from_file_location(
        "vllm.vllm_flash_attn",
        f"{source_dir}/__init__.py",
        submodule_search_locations=[source_dir],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm.vllm_flash_attn"] = module
    spec.loader.exec_module(module)


_preload_local_vllm_flash_attn()


def _patch_omni_init_timeouts() -> None:
    init_timeout = os.environ.get("WAN_OMNI_INIT_TIMEOUT")
    stage_init_timeout = os.environ.get("WAN_OMNI_STAGE_INIT_TIMEOUT")
    if not init_timeout and not stage_init_timeout:
        return

    from vllm_omni.entrypoints.omni_base import OmniBase

    original_init = OmniBase.__init__

    def patched_init(self, *args, **kwargs):
        if init_timeout:
            kwargs.setdefault("init_timeout", int(init_timeout))
        if stage_init_timeout:
            kwargs.setdefault("stage_init_timeout", int(stage_init_timeout))
        return original_init(self, *args, **kwargs)

    OmniBase.__init__ = patched_init


_patch_omni_init_timeouts()


def _patch_omni_async_output_timeout() -> None:
    async_output_timeout = os.environ.get("WAN_OMNI_ASYNC_OUTPUT_TIMEOUT")
    if not async_output_timeout:
        return

    from vllm_omni.diffusion import diffusion_engine

    diffusion_engine._ASYNC_OUTPUT_TIMEOUT = float(async_output_timeout)


_patch_omni_async_output_timeout()
