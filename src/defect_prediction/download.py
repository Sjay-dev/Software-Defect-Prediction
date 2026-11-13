"""Download the immutable Zenodo release files and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(manifest_path: Path, output_dir: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    reused = 0
    for record in manifest["records"]:
        record_id = record["zenodo_record"]
        for filename, expected_md5 in record["files"].items():
            destination = output_dir / filename
            if destination.exists() and _md5(destination) == expected_md5:
                reused += 1
                continue
            if destination.exists():
                raise RuntimeError(
                    f"Checksum mismatch for existing {destination}. Move it aside and retry."
                )
            url = (
                f"https://zenodo.org/api/records/{record_id}/files/"
                f"{urllib.parse.quote(filename)}/content"
            )
            temporary = destination.with_suffix(destination.suffix + ".part")
            request = urllib.request.Request(
                url, headers={"User-Agent": "software-defect-prediction/0.1"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(response, target)
                actual_md5 = _md5(temporary)
                if actual_md5 != expected_md5:
                    raise RuntimeError(
                        f"Checksum mismatch for {filename}: expected {expected_md5}, "
                        f"got {actual_md5}."
                    )
                temporary.replace(destination)
                downloaded += 1
                print(f"Downloaded {filename}")
            finally:
                if temporary.exists():
                    temporary.unlink()
    print(f"Dataset ready: {downloaded} downloaded, {reused} already verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/data_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download_dataset(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
