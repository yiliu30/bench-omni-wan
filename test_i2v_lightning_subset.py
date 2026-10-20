import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import cv2
import numpy as np
import pytest

import i2v_lightning_subset as runner
from compare_i2v_lightning_subset import compare, ssim


def fixture_config(tmp_path: Path, count: int = 2) -> dict[str, str]:
    metadata = tmp_path / "info.json"
    metadata.write_text(json.dumps([
        {"prompt_en": "not imaging", "dimension": ["camera_motion"], "image_name": "other.jpg"},
        *({"prompt_en": f"image {index}", "dimension": ["imaging_quality"],
           "image_name": f"crop{index}.jpg", "image_type": "abstract"}
          for index in range(count + 1))
    ]))
    return {"ARCHIVE": str(tmp_path / "official.zip"), "METADATA": str(metadata),
            "CROPS": str(tmp_path / "official_crop" / "1-1"), "OUTPUT": str(tmp_path / "output"),
            "COUNT": str(count), "MODEL": "/fake/model", "PORT_BASE": "18600",
            "PYTHON": "/fake/python", "VLLM_OMNI_ROOT": "/fake/repo",
            "DEEPKLOX_REPO": "/fake/deepklox", "INIT_TIMEOUT": "2400",
            "VAE_FAST_DECODE": "0"}


def test_first_matching_rows_in_upstream_order(tmp_path):
    cfg = fixture_config(tmp_path)
    selected = runner.selected_rows(Path(cfg["METADATA"]), 2)
    assert [row["prompt_en"] for row in selected] == ["image 0", "image 1"]
    assert runner.selected_rows(Path(cfg["METADATA"]), 3)[-1]["prompt_en"] == "image 2"


def test_official_archive_required(tmp_path):
    cfg = fixture_config(tmp_path)
    with patch.object(runner, "UPSTREAM_METADATA_SHA256", runner.digest(Path(cfg["METADATA"]))):
        with pytest.raises(FileNotFoundError, match="Official crop.zip"):
            runner.extract_official(cfg)
    with pytest.raises(FileNotFoundError, match="provenance missing"):
        with patch.object(runner, "UPSTREAM_METADATA_SHA256", runner.digest(Path(cfg["METADATA"]))):
            runner.prepare(cfg)


def test_extract_and_freeze_manifest(tmp_path):
    cfg = fixture_config(tmp_path)
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 48, 3), 110, np.uint8))
    assert ok
    with zipfile.ZipFile(cfg["ARCHIVE"], "w") as archive:
        for index in range(2):
            archive.writestr(f"crop/1-1/crop{index}.jpg", jpeg.tobytes())
    with patch.object(runner, "UPSTREAM_METADATA_SHA256", runner.digest(Path(cfg["METADATA"]))):
        runner.extract_official(cfg)
        manifest = runner.prepare(cfg)
        assert [(row["index"], row["device"], row["stem"]) for row in manifest["items"]] == [
            (0, 0, "000"), (1, 1, "001")
        ]
        assert runner.prepare(cfg) == manifest
        crop = Path(cfg["CROPS"]) / "crop0.jpg"
        crop.write_bytes(b"not an official crop")
        with pytest.raises(ValueError, match="checksum mismatch"):
            runner.prepare(cfg)


def test_missing_archive_member_rejected(tmp_path):
    cfg = fixture_config(tmp_path)
    with zipfile.ZipFile(cfg["ARCHIVE"], "w") as archive:
        archive.writestr("crop/1-1/crop0.jpg", b"input")
    with patch.object(runner, "UPSTREAM_METADATA_SHA256", runner.digest(Path(cfg["METADATA"]))):
        with pytest.raises(ValueError, match="missing 1 selected crop"):
            runner.extract_official(cfg)


def test_direct_official_folder_provenance(tmp_path):
    cfg = fixture_config(tmp_path)
    crops = Path(cfg["CROPS"])
    crops.mkdir(parents=True)
    ok, jpeg = cv2.imencode(".jpg", np.full((48, 48, 3), 110, np.uint8))
    assert ok
    for index in range(2):
        (crops / f"crop{index}.jpg").write_bytes(jpeg.tobytes())
    source = {"source": f"https://drive.google.com/drive/folders/{runner.OFFICIAL_FOLDER_ID}",
              "metadata_sha256": runner.digest(Path(cfg["METADATA"])),
              "upstream_commit": runner.UPSTREAM_REVISION, "count": 2,
              "file_ids": {"crop0.jpg": "drive-id-0", "crop1.jpg": "drive-id-1"},
              "crop_sha256": {f"crop{index}.jpg": runner.digest(crops / f"crop{index}.jpg")
                              for index in range(2)}}
    with patch.object(runner, "UPSTREAM_METADATA_SHA256", source["metadata_sha256"]):
        runner.write_json(crops.parent / "source.json", source)
        manifest = runner.prepare(cfg)
        assert manifest["official_file_ids"] == source["file_ids"]
        source["file_ids"].pop("crop0.jpg")
        runner.write_json(crops.parent / "source.json", source)
        with pytest.raises(ValueError, match="do not cover"):
            runner.prepare(cfg)


def test_request_matches_paired_model_configuration(tmp_path):
    cfg = runner.config(Path(__file__).with_name("i2v_lightning_50.env"))
    image = tmp_path / "crop0.jpg"
    image.write_bytes(b"fake jpeg")
    item = {"image": str(image), "prompt": "a blue car"}
    body, content_type = runner.video_request(item)
    assert content_type.startswith("multipart/form-data; boundary=")
    for value in (b'name="input_reference"', b'name="extra_params"', b'"sample_solver": "euler"',
                  b'name="boundary_ratio"', b"0.9", b'name="flow_shift"', b'name="num_frames"', b"81",
                  b"fake jpeg", b"a blue car"):
        assert value in body
    command = runner.server_command(cfg, 2, "mxfp4_sage_flash4")
    assert command[command.index("--quantization") + 1] == "mxfp4"
    assert "--enforce-eager" in command
    assert "--quantization" not in runner.server_command(cfg, 2, "bf16_flash")
    assert runner.server_env(cfg, 2, "bf16_flash")["SAGE_ATTN_FORCE_SDPA_BLOCKS"] == runner.ALL_BLOCKS
    assert runner.server_env(cfg, 2, "mxfp4_sage_flash4")["SAGE_ATTN_FORCE_SDPA_BLOCKS"] == "0,33,34,38"
    assert runner.server_env(cfg, 2, "bf16_flash")["VLLM_OMNI_WAN_VAE_FAST_DECODE"] == "0"


def test_recipe_envs_reject_changed_recipe_or_shared_settings(tmp_path):
    cfg = runner.config(Path(__file__).with_name("i2v_lightning_50.env"))
    with pytest.raises(ValueError, match="paired runner"):
        runner.config(runner.RECIPE_ENV_FILES["bf16_flash"])
    assert runner.recipe_config(cfg, "bf16_flash")["QUANTIZATION"] == "none"
    assert runner.recipe_config(cfg, "mxfp4_sage_flash4")["QUANTIZATION"] == "mxfp4"
    original = runner.RECIPE_ENV_FILES["mxfp4_sage_flash4"].read_text()
    modified = tmp_path / "sage.env"
    with patch.dict(runner.RECIPE_ENV_FILES, {"mxfp4_sage_flash4": modified}):
        modified.write_text(original.replace("QUANTIZATION=mxfp4", "QUANTIZATION=none"))
        with pytest.raises(ValueError, match="QUANTIZATION differs"):
            runner.config(Path(__file__).with_name("i2v_lightning_50.env"))
        modified.write_text(original.replace("SEED=0", "SEED=1"))
        with pytest.raises(ValueError, match="SEED differs"):
            runner.config(Path(__file__).with_name("i2v_lightning_50.env"))


def test_no_fallback_env_keeps_settings_and_disables_forced_blocks(tmp_path):
    path = runner.RECIPE_ENV_FILES[runner.NO_FALLBACK_RECIPE]
    cfg = runner.config(path)
    baseline = runner.config(Path(__file__).with_name("i2v_lightning_50.env"))
    assert cfg["OUTPUT"] != baseline["OUTPUT"]
    assert cfg["FALLBACK_BLOCKS"] == ""
    assert {key: value for key, value in cfg.items() if key not in ("OUTPUT", "FALLBACK_BLOCKS")} == {
        key: value for key, value in baseline.items() if key != "OUTPUT"
    }
    assert runner.recipe_config(cfg, runner.NO_FALLBACK_RECIPE)["QUANTIZATION"] == "mxfp4"
    assert runner.server_env(cfg, 2, runner.NO_FALLBACK_RECIPE)["SAGE_ATTN_FORCE_SDPA_BLOCKS"] == ""
    assert runner.server_command(cfg, 2, runner.NO_FALLBACK_RECIPE)[-2:] == ["--quantization", "mxfp4"]
    altered = tmp_path / "no_fallback.env"
    with patch.dict(runner.RECIPE_ENV_FILES, {runner.NO_FALLBACK_RECIPE: altered}):
        altered.write_text(path.read_text().replace("SAGE_ATTN_FORCE_SDPA_BLOCKS=\n",
                                                   "SAGE_ATTN_FORCE_SDPA_BLOCKS=0\n"))
        with pytest.raises(ValueError, match="SAGE_ATTN_FORCE_SDPA_BLOCKS differs"):
            runner.config(altered)
        altered.write_text(path.read_text().replace("SEED=0", "SEED=1"))
        with pytest.raises(ValueError, match="must match the paired campaign"):
            runner.config(altered)


def test_reject_wrong_frame_count(tmp_path):
    path = tmp_path / "bad.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (640, 640))
    assert writer.isOpened()
    writer.write(np.full((640, 640, 3), 100, np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="Expected 81 frames"):
        runner.validate_video(path)


def test_ssim_identical_and_changed():
    frame = np.full((64, 64, 3), 95, np.uint8)
    changed = frame.copy()
    changed[:, 32:] = 170
    assert ssim(frame, frame) == pytest.approx(1.0, abs=1e-5)
    assert ssim(frame, changed) < 1.0


def test_routing_counters_require_real_sage_and_flash(tmp_path):
    path = tmp_path / "server.log"
    path.write_text("Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0\n"
                    "Sage attention V3 call counts: sage=180, forced_sdpa=20, "
                    "cross_sdpa=200, sdpa_fallback=0, nonfinite_sdpa=0, fallback_flash=20\n")
    runner.verify_routing(path, "mxfp4_sage_flash4")
    with pytest.raises(RuntimeError, match="Unexpected Sage routing"):
        runner.verify_routing(path, "bf16_flash")
    path.write_text("Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0\n"
                    "Sage attention V3 call counts: sage=0, forced_sdpa=200, "
                    "cross_sdpa=200, sdpa_fallback=0, nonfinite_sdpa=0, fallback_flash=200\n")
    runner.verify_routing(path, "bf16_flash")
    path.write_text("Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0\n"
                    "server stopped before counters")
    with pytest.raises(RuntimeError, match="counters missing"):
        runner.verify_routing(path, "mxfp4_sage_flash4")
    path.write_text("Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0\n"
                    "Sage attention V3 call counts: sage=200, forced_sdpa=0, "
                    "cross_sdpa=200, sdpa_fallback=0, nonfinite_sdpa=0, fallback_flash=0\n")
    runner.verify_routing(path, runner.NO_FALLBACK_RECIPE)
    with pytest.raises(RuntimeError, match="Unexpected Sage routing"):
        runner.verify_routing(path, "mxfp4_sage_flash4")


def test_resume_verified_worker_without_reloading_server(tmp_path):
    cfg = fixture_config(tmp_path)
    item = {"stem": "000", "index": 0, "device": 0, "sha256": "image",
            "prompt": "sunset"}
    manifest = {"metadata_sha256": "metadata", "items": [item]}
    root = Path(cfg["OUTPUT"]) / "bf16_flash"
    root.mkdir(parents=True)
    (root / "000.mp4").write_bytes(b"video")
    runner.write_json(root / "000.json", {
        "status": "completed", "recipe": "bf16_flash", "index": 0, "device": 0,
        "image_sha256": "image", "metadata_sha256": "metadata", "prompt": "sunset",
        "video": {"sha256": "verified"},
    })
    (root / "server_xpu0.log").write_text(
        "Wan VAE fast decode disabled by VLLM_OMNI_WAN_VAE_FAST_DECODE=0\n"
        "Sage attention V3 call counts: sage=0, forced_sdpa=200, "
        "cross_sdpa=200, sdpa_fallback=0, nonfinite_sdpa=0, fallback_flash=200\n"
    )
    with patch.object(runner, "validate_video", return_value={"sha256": "verified"}), \
            patch.object(runner.socket, "socket", side_effect=AssertionError("server restarted")):
        runner.serve_and_run(cfg, manifest, 0, "bf16_flash")


def test_paired_comparison_writes_expected_outputs(tmp_path):
    manifest = {"metadata_sha256": "metadata", "items": [
        {"stem": "000", "index": 0, "prompt": "sunset", "image_name": "sunset.jpg",
         "image_type": "scenery", "sha256": "image"}
    ]}
    runner.write_json(tmp_path / "manifest.json", manifest)
    for recipe, offset in (("bf16_flash", 0), ("mxfp4_sage_flash4", 10)):
        directory = tmp_path / recipe
        directory.mkdir()
        video = directory / "000.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 16, (640, 640))
        assert writer.isOpened()
        y, x = np.indices((640, 640))
        pattern = (75 + (x + y) % 40).astype(np.uint8)
        for index in range(81):
            writer.write(cv2.cvtColor(pattern + index // 5 + offset, cv2.COLOR_GRAY2BGR))
        writer.release()
        runner.write_json(directory / "000.json", {
            "status": "completed", "recipe": recipe, "prompt": "sunset", "index": 0,
            "image_sha256": "image", "metadata_sha256": "metadata",
            "video": runner.validate_video(video),
        })
    summary = compare(tmp_path)
    assert summary["pairs"] == 1
    assert summary["mean_mae_0_255"] == pytest.approx(10, abs=0.1)
    assert summary["mean_ssim"] < 1
    assert len(list((tmp_path / "comparison").glob("pair_*.png"))) == 5
    assert len((tmp_path / "comparison/per_frame.csv").read_text().splitlines()) == 82
