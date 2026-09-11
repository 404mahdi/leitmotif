"""Download raw datasets into data/raw.

    python scripts/download_data.py musiccaps-csv   # MusicCaps captions + aspect tags (a few MB)
    python scripts/download_data.py fma-small       # FMA metadata + fma_small audio (~7.6 GB)

Uses only the standard library, so it runs before the environment is set up.
Interrupted downloads resume from where they stopped.
"""

import argparse
import hashlib
import http.client
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

MUSICCAPS_CSV = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"

FMA_URL = "https://os.unil.cloud.switch.ch/fma/"
# SHA-1 checksums of the current release, from github.com/mdeff/fma
FMA_ARCHIVES = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
}

CHUNK = 1 << 20
REPORT_EVERY = 256 * CHUNK


def _size(n: int) -> str:
    return f"{n / (1 << 30):.2f} GiB" if n >= 1 << 30 else f"{n / (1 << 20):.1f} MiB"


def _download_once(url: str, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as err:
        if err.code != 416:  # 416 means the .part file already holds the whole file
            raise
    else:
        with response:
            if done and response.status != 206:  # server ignored the Range header
                done = 0
            total = done + int(response.headers.get("Content-Length", 0))
            state = f"resuming at {_size(done)}" if done else "starting"
            print(f"{dest.name}: {state} of {_size(total)}", flush=True)
            next_report = done + REPORT_EVERY
            with open(part, "ab" if done else "wb") as f:
                while chunk := response.read(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        print(f"  {_size(done)} / {_size(total)}", flush=True)
                        next_report += REPORT_EVERY
    part.replace(dest)


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream url to dest through a .part file, retrying and resuming on network errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        try:
            return _download_once(url, dest)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as err:
            if isinstance(err, urllib.error.HTTPError) or attempt == attempts:
                raise
            print(f"  {err!r}; retrying in {10 * attempt} s", flush=True)
            time.sleep(10 * attempt)


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(16 * CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def musiccaps_csv() -> None:
    dest = RAW / "musiccaps" / "musiccaps-public.csv"
    if dest.exists():
        print(f"{dest.relative_to(ROOT)} exists, skipping")
        return
    download(MUSICCAPS_CSV, dest)
    print(f"Saved {dest.relative_to(ROOT)}")


def fma_small() -> None:
    target = RAW / "fma"
    for name, expected in FMA_ARCHIVES.items():
        folder = target / name.removesuffix(".zip")
        if folder.exists():
            print(f"{folder.relative_to(ROOT)} exists, skipping")
            continue
        archive = target / name
        if not archive.exists():
            download(FMA_URL + name, archive)
        print(f"Verifying {name}", flush=True)
        actual = sha1(archive)
        if actual != expected:
            sys.exit(f"{name}: SHA-1 is {actual}, expected {expected}. Delete the file and run again.")
        # Extract into a staging folder so an interrupted run never leaves a half-filled folder
        staging = target / f"_extracting_{folder.name}"
        print(f"Extracting {name}", flush=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
        inner = staging / folder.name
        if inner.is_dir():
            inner.replace(folder)
            staging.rmdir()
        else:
            staging.replace(folder)
        archive.unlink()
        print(f"Ready: {folder.relative_to(ROOT)}", flush=True)


DATASETS = {"musiccaps-csv": musiccaps_csv, "fma-small": fma_small}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", choices=DATASETS)
    for name in parser.parse_args().datasets:
        DATASETS[name]()


if __name__ == "__main__":
    main()
