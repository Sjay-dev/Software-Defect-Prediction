from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from defect_prediction.download import download_dataset


def _manifest(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "zenodo_record": 123,
                        "files": {
                            "sample.csv": hashlib.md5(
                                content, usedforsecurity=False
                            ).hexdigest()
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://zenodo.org/test", code, "test failure", None, None
    )


def test_download_retries_transient_http_errors(monkeypatch, tmp_path, capsys):
    content = b"column\nvalue\n"
    manifest = _manifest(tmp_path, content)
    calls = 0

    def urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 60
        if calls < 3:
            raise _http_error(500)
        return io.BytesIO(content)

    delays: list[float] = []
    monkeypatch.setattr("defect_prediction.download.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("defect_prediction.download.time.sleep", delays.append)

    output_dir = tmp_path / "raw"
    download_dataset(manifest, output_dir)

    assert calls == 3
    assert delays == [1.0, 2.0]
    assert (output_dir / "sample.csv").read_bytes() == content
    assert not (output_dir / "sample.csv.part").exists()
    assert "Retrying in 1s (attempt 2/5)" in capsys.readouterr().err


def test_download_does_not_retry_permanent_http_errors(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, b"content")
    calls = 0

    def urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(404)

    monkeypatch.setattr("defect_prediction.download.urllib.request.urlopen", urlopen)

    with pytest.raises(urllib.error.HTTPError, match="HTTP Error 404"):
        download_dataset(manifest, tmp_path / "raw", initial_retry_delay=0)

    assert calls == 1


def test_download_reports_exhausted_transient_retries(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, b"content")
    calls = 0

    def urlopen(_request, timeout):
        nonlocal calls
        calls += 1
        raise _http_error(503)

    monkeypatch.setattr("defect_prediction.download.urllib.request.urlopen", urlopen)

    with pytest.raises(RuntimeError, match="after 3 attempts: HTTP 503"):
        download_dataset(
            manifest,
            tmp_path / "raw",
            max_attempts=3,
            initial_retry_delay=0,
        )

    assert calls == 3
    assert not (tmp_path / "raw" / "sample.csv.part").exists()
