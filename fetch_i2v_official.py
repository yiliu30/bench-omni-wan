#!/usr/bin/env python3
"""Download only the selected original VBench 1:1 crops from its official Drive folder."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import uuid

import gdown

from run import load_env


FOLDER_ID = "1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx"
REVISION = "fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490"
METADATA_HASH = "8cf4d34dc11779a336c216a1a71d482927cfa24a1bf736ab5c92425609f129f8"


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save(path: Path, data: dict) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch(cfg: dict[str, str]) -> None:
    metadata = Path(cfg["METADATA"])
    if checksum(metadata) != METADATA_HASH:
        raise ValueError("Metadata differs from pinned VBench revision")
    rows = [row for row in json.loads(metadata.read_text(encoding="utf-8"))
            if "imaging_quality" in row["dimension"]][:int(cfg["COUNT"])]
    names = [row["image_name"] for row in rows]
    if len(names) != int(cfg["COUNT"]) or len(set(names)) != len(names):
        raise ValueError("Expected distinct first 50 imaging_quality images")
    files = gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True, timeout=40)
    if files is None:
        raise RuntimeError("Cannot list official VBench Google Drive folder")
    targets = {}
    for file in files:
        if file.path.startswith("crop/1-1/") and file.path.removeprefix("crop/1-1/") in names:
            name = file.path.removeprefix("crop/1-1/")
            if name in targets:
                raise ValueError(f"Duplicate official crop in Drive listing: {name}")
            targets[name] = file.id
    if set(targets) != set(names):
        raise ValueError(f"Official Drive folder missing {sorted(set(names) - set(targets))}")
    root = Path(cfg["CROPS"])
    source = root.parent / "source.json"
    state_path = root.parent / "download_state.json"
    if source.exists():
        raise FileExistsError(f"Provenance already exists; refusing to overwrite: {source}")
    root.mkdir(parents=True, exist_ok=True)
    state = {"source": f"https://drive.google.com/drive/folders/{FOLDER_ID}",
             "metadata_sha256": METADATA_HASH, "upstream_commit": REVISION,
             "count": len(names), "file_ids": targets, "crop_sha256": {}}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if {**previous, "crop_sha256": {}} != state:
            raise ValueError(f"Download state does not match official folder listing: {state_path}")
        state = previous
    for name in names:
        destination = root / name
        if name in state["crop_sha256"]:
            if checksum(destination) != state["crop_sha256"][name]:
                raise ValueError(f"Previously downloaded crop changed: {destination}")
            continue
        if destination.exists():
            raise FileExistsError(f"Untracked crop, refusing to replace: {destination}")
        part = root / f".{uuid.uuid4().hex}.part"
        try:
            for attempt in range(3):
                try:
                    try:
                        result = gdown.download(id=targets[name], output=str(part), quiet=True,
                                                timeout=45, retries=1)
                    except gdown.exceptions.FileURLRetrievalError:
                        direct = (f"https://drive.usercontent.google.com/download"
                                  f"?id={targets[name]}&export=download")
                        with urllib.request.urlopen(direct, timeout=60) as response:
                            if response.status != 200 or response.headers.get_content_type() != "image/jpeg":
                                raise RuntimeError(f"Official Drive file is not a JPEG: {name}")
                            with part.open("wb") as sink:
                                shutil.copyfileobj(response, sink)
                        result = str(part)
                    if not result or not part.is_file() or part.stat().st_size < 1024:
                        raise RuntimeError(f"Official crop download did not return a JPEG: {name}")
                    with part.open("rb") as stream:
                        if stream.read(3) != b"\xff\xd8\xff":
                            raise ValueError(f"Unexpected official crop file format: {name}")
                    break
                except (OSError, RuntimeError, ValueError, urllib.error.URLError,
                        gdown.exceptions.FileURLRetrievalError):
                    if attempt == 2:
                        raise
                    part.unlink(missing_ok=True)
                    time.sleep(5)
            part.replace(destination)
            state["crop_sha256"][name] = checksum(destination)
            save(state_path, state)
            print(f"Downloaded {len(state['crop_sha256'])}/{len(names)}: {name}", flush=True)
        finally:
            part.unlink(missing_ok=True)
    save(source, state)
    print(f"All {len(names)} official crops downloaded; provenance: {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env_file", nargs="?", default="i2v_lightning_50.env")
    args = parser.parse_args()
    path = Path(args.env_file)
    cfg, _ = load_env(str(path if path.is_absolute() else Path(__file__).parent / path))
    fetch(cfg)
