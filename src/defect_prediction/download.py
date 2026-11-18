"""Download the immutable Zenodo release files and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


RETRYABLE_HTTP_STATUSES = {408, 429}
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_INITIAL_RETRY_DELAY = 1.0


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_with_retries(
    request: urllib.request.Request,
    temporary: Path,
    filename: str,
    *,
    max_attempts: int,
    initial_retry_delay: float,
) -> None:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if initial_retry_delay < 0:
        raise ValueError("initial_retry_delay cannot be negative.")

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
            return
        except urllib.error.HTTPError as error:
            caught_error = error
            retryable = error.code in RETRYABLE_HTTP_STATUSES or 500 <= error.code < 600
            reason = f"HTTP {error.code}"
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as error:
            caught_error = error
            retryable = True
            reason = str(error.reason if isinstance(error, urllib.error.URLError) else error)

        if temporary.exists():
            temporary.unlink()
        if not retryable:
            raise caught_error
        if attempt == max_attempts:
            raise RuntimeError(
                f"Failed to download {filename} after {max_attempts} attempts: {reason}."
            ) from caught_error

        delay = initial_retry_delay * (2 ** (attempt - 1))
        print(
            f"Transient download failure for {filename}: {reason}. "
            f"Retrying in {delay:g}s (attempt {attempt + 1}/{max_attempts}).",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)


def download_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_retry_delay: float = DEFAULT_INITIAL_RETRY_DELAY,
) -> None:
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
                _download_with_retries(
                    request,
                    temporary,
                    filename,
                    max_attempts=max_attempts,
                    initial_retry_delay=initial_retry_delay,
                )
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
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-retry-delay", type=float, default=DEFAULT_INITIAL_RETRY_DELAY
    )
    args = parser.parse_args()
    download_dataset(
        args.manifest,
        args.output_dir,
        max_attempts=args.max_attempts,
        initial_retry_delay=args.initial_retry_delay,
    )


if __name__ == "__main__":
    main()
