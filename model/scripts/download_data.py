#!/usr/bin/env python3
"""Download the PPG-DaLiA dataset into ``DATA_DIR``.

PPG-DaLiA is hosted on the UCI ML Repository (dataset #495) under CC BY 4.0. It is
~2.6 GB and is NEVER committed (see .gitignore). This script fetches the archive,
extracts it (the UCI download nests a ``PPG_FieldStudy.zip`` inside the outer zip),
and leaves ``DATA_DIR/PPG_FieldStudy/S{1..15}/S{n}.pkl`` on disk.

    Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
    Cite:   A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven,
            "Deep PPG: Large-scale Heart Rate Estimation with Convolutional
            Neural Networks", Sensors, 2019.

Network/TLS note: the download uses ``urllib`` and honours the standard
``HTTPS_PROXY`` and ``SSL_CERT_FILE``/``REQUESTS_CA_BUNDLE`` environment variables,
so it works both directly and behind a TLS-inspecting proxy.

Educational prototype — NOT a medical device.
"""
from __future__ import annotations

import argparse
import os
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

PPG_DALIA_PAGE = "https://archive.ics.uci.edu/dataset/495/ppg+dalia"
DEFAULT_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
DATASET_DIRNAME = "PPG_FieldStudy"
NUM_SUBJECTS = 15


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, picking up a custom CA bundle if the env provides one."""
    ca = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
    )
    if ca and Path(ca).exists():
        return ssl.create_default_context(cafile=ca)
    return ssl.create_default_context()


def resolve_data_dir(cli_value: str | None) -> Path:
    """DATA_DIR precedence: --data-dir > $DATA_DIR > ./data (matches .env.example)."""
    raw = cli_value or os.environ.get("DATA_DIR") or "./data"
    return Path(raw).expanduser().resolve()


def subject_pickles(data_dir: Path) -> list[Path]:
    """Return the present ``S{n}/S{n}.pkl`` paths under ``PPG_FieldStudy``."""
    root = data_dir / DATASET_DIRNAME
    return [p for n in range(1, NUM_SUBJECTS + 1) if (p := root / f"S{n}" / f"S{n}.pkl").exists()]


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url}\n       ->  {dest}")
    ctx = _ssl_context()
    req = urllib.request.Request(url, headers={"User-Agent": "biosignal-model/0.1 (+educational)"})
    with urllib.request.urlopen(req, context=ctx) as resp:  # noqa: S310 (trusted UCI URL)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1 << 20  # 1 MiB
        next_report = 0
        with open(dest, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if done >= next_report:
                    mb = done / 1e6
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    print(f"  … {mb:,.0f} MB{pct}", flush=True)
                    next_report = done + 100 * (1 << 20)  # every ~100 MB
    print(f"  done: {dest.stat().st_size / 1e6:,.0f} MB")


def _extract_recursive(archive: Path, dest: Path, *, remove_nested: bool) -> None:
    """Extract ``archive`` into ``dest``, then extract any zips it produced (one pass deep)."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    nested = [p for p in dest.rglob("*.zip") if p != archive]
    for inner in nested:
        print(f"Extracting nested archive {inner.name}")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
        if remove_nested:
            inner.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the PPG-DaLiA dataset into DATA_DIR.")
    parser.add_argument("--data-dir", default=None, help="Target directory (default: $DATA_DIR or ./data).")
    parser.add_argument("--url", default=os.environ.get("PPG_DALIA_URL", DEFAULT_URL), help="Archive URL.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the data is present.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip after extraction.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    present = subject_pickles(data_dir)
    if len(present) == NUM_SUBJECTS and not args.force:
        print(f"PPG-DaLiA already present: {len(present)}/{NUM_SUBJECTS} subjects in {data_dir/DATASET_DIRNAME}")
        print("Use --force to re-download.")
        return

    zip_path = data_dir / "ppg+dalia.zip"
    try:
        _download(args.url, zip_path)
        print(f"Extracting into {data_dir} …")
        _extract_recursive(zip_path, data_dir, remove_nested=not args.keep_zip)
    except Exception as exc:  # surface a clear, actionable message
        raise SystemExit(
            f"Download/extraction failed: {exc}\n"
            f"See {PPG_DALIA_PAGE} (CC BY 4.0, ~2.6 GB). "
            f"You can also fetch it manually and unzip into {data_dir}."
        ) from exc
    finally:
        if zip_path.exists() and not args.keep_zip:
            zip_path.unlink()

    present = subject_pickles(data_dir)
    if not present:
        raise SystemExit(
            "Extraction finished but no S{n}/S{n}.pkl files were found. "
            f"Inspect {data_dir} — the archive layout may have changed."
        )
    print(f"\nReady: {len(present)}/{NUM_SUBJECTS} subject pickles under {data_dir/DATASET_DIRNAME}")
    if len(present) < NUM_SUBJECTS:
        print("Warning: some subjects are missing; check the extraction above.")


if __name__ == "__main__":
    main()
