from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CJKVI_IDS_REVISION = "86b4d16159f0079437870408f0ca186e529015db"
CJKVI_IDS_URL = (
    "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/"
    f"{CJKVI_IDS_REVISION}/ids.txt"
)
CJKVI_IDS_SHA256 = "bfc70a8c09f9f5616ebf0543bd6681e67314e9f7ae2307e5ae8c6f15bdc5c6a6"
DEFAULT_IDS_PATH = "data/cjkvi-ids/ids.txt"


class IDSDataError(RuntimeError):
    """Raised when optional CJKVI IDS data cannot be installed safely."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (_project_root() / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source_metadata(destination: Path, *, url: str, sha256: str) -> Path:
    metadata_path = destination.with_name("source.json")
    payload = {
        "project": "cjkvi/cjkvi-ids",
        "upstream": "https://github.com/cjkvi/cjkvi-ids",
        "file": "ids.txt",
        "revision": CJKVI_IDS_REVISION,
        "download_url": url,
        "sha256": sha256,
        "bundled_by_hanzistyleforge": False,
        "license_note": "ids.txt is derived from CHISE; its license follows the upstream CHISE/CJKVI terms.",
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def install_cjkvi_ids(
    destination: str | Path = DEFAULT_IDS_PATH,
    *,
    force: bool = False,
    url: str = CJKVI_IDS_URL,
    expected_sha256: str = CJKVI_IDS_SHA256,
    timeout_seconds: int = 90,
    retries: int = 3,
) -> dict[str, Any]:
    """Download the pinned upstream ids.txt directly into the user's project.

    HanziStyleForge does not redistribute this data file. The downloader uses a
    pinned upstream revision and verifies the downloaded bytes before replacing
    the destination atomically.
    """

    target = resolve_project_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = str(expected_sha256).strip().lower()
    if target.is_file() and not force:
        actual = _sha256(target)
        return {
            "installed": False,
            "reason": "already present",
            "path": str(target),
            "sha256": actual,
            "matches_pinned_revision": bool(expected and actual == expected),
        }

    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        temporary_path: Path | None = None
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "HanziStyleForge/2.3 CJKVI-IDS installer"},
            )
            with urllib.request.urlopen(request, timeout=max(10, int(timeout_seconds))) as response:
                with tempfile.NamedTemporaryFile(
                    prefix="ids-", suffix=".txt.tmp", dir=target.parent, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        temporary.write(chunk)
            actual = _sha256(temporary_path)
            if expected and actual != expected:
                raise IDSDataError(
                    "Downloaded CJKVI ids.txt failed SHA-256 verification: "
                    f"expected {expected}, received {actual}."
                )
            os.replace(temporary_path, target)
            metadata = _write_source_metadata(target, url=url, sha256=actual)
            return {
                "installed": True,
                "path": str(target),
                "source_metadata": str(metadata),
                "url": url,
                "revision": CJKVI_IDS_REVISION,
                "sha256": actual,
            }
        except (OSError, urllib.error.URLError, IDSDataError) as exc:
            last_error = exc
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < max(1, int(retries)):
                time.sleep(min(8.0, 1.5 * attempt))
    raise IDSDataError(
        "Unable to download the optional CJKVI IDS file. Run the ids-install "
        "command again when internet access is available, or place ids.txt at "
        f"{target}. Last error: {type(last_error).__name__}: {last_error}"
    )


def ensure_decomposition_data(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve_project_path(config.get("decomposition_file", DEFAULT_IDS_PATH))
    if path.is_file():
        actual = _sha256(path)
        expected = str(config.get("source_sha256", CJKVI_IDS_SHA256)).strip().lower()
        return path, {
            "available": True,
            "downloaded": False,
            "path": str(path),
            "sha256": actual,
            "matches_pinned_revision": bool(expected and actual == expected),
        }
    if not bool(config.get("auto_download", True)):
        return path, {
            "available": False,
            "downloaded": False,
            "path": str(path),
            "reason": "auto-download disabled",
        }
    result = install_cjkvi_ids(
        path,
        url=str(config.get("source_url", CJKVI_IDS_URL)),
        expected_sha256=str(config.get("source_sha256", CJKVI_IDS_SHA256)),
    )
    return path, {"available": True, "downloaded": True, **result}
