"""Download the 10-second MusicCaps clips from YouTube as 22,050 Hz mono WAV.

    uv run python scripts/download_musiccaps_audio.py              # all 5,521 clips
    uv run python scripts/download_musiccaps_audio.py --limit 20   # quick test

Needs ffmpeg and a JavaScript runtime (Node, Deno or Bun) on PATH, which
yt-dlp uses for YouTube. Finished clips are skipped, so re-running resumes.
Videos that are private, removed or otherwise gone are listed in
data/raw/musiccaps/failed.csv and skipped on later runs.
"""

import argparse
import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MUSICCAPS = ROOT / "data" / "raw" / "musiccaps"
AUDIO = MUSICCAPS / "audio"
STAGING = AUDIO / "_staging"
FAILED = MUSICCAPS / "failed.csv"
SAMPLE_RATE = 22050

# Error messages meaning the video itself is gone, so retrying won't help
PERMANENT = (
    "video unavailable",
    "video is unavailable",
    "private video",
    "has been removed",
    "account associated with this video has been terminated",
    "not available in your country",
    "confirm your age",
    "members-only",
    "copyright",
    "clip too short",
)
# Error messages meaning YouTube is throttling us rather than the video being missing
BLOCKED = ("not a bot", "http error 429")


def kill_tree(process: subprocess.Popen) -> None:
    """Stop yt-dlp together with the ffmpeg process it started."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGKILL)


def remove_staged(ytid: str) -> None:
    for leftover in STAGING.glob(f"{ytid}.*"):
        try:
            leftover.unlink(missing_ok=True)
        except PermissionError:  # still held by a process that is shutting down; cleared on the next run
            pass


def download_clip(ytid: str, start: int, end: int, args: argparse.Namespace) -> tuple[str, str | None]:
    """Fetch one clip into the staging folder, check its length, then move it into place."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings", "--no-progress",
        "--js-runtimes", args.js_runtime,
        "--cache-dir", str(ROOT / ".cache" / "yt-dlp"),
        "-f", "bestaudio/best",
        "--download-sections", f"*{start}-{end}",
        # Re-encode at the cut points; a plain stream copy snaps to ~10 s chunks and grabs the wrong audio
        "--force-keyframes-at-cuts",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ExtractAudio:-ar {SAMPLE_RATE} -ac 1",
        "-o", str(STAGING / "%(id)s.%(ext)s"),
    ]
    if args.cookies_from_browser:
        cmd += ["--cookies-from-browser", args.cookies_from_browser]
    cmd.append(f"https://www.youtube.com/watch?v={ytid}")

    staged = STAGING / f"{ytid}.wav"
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        start_new_session=sys.platform != "win32",
    )
    try:
        _, stderr = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        # A stalled connection can leave ffmpeg waiting forever
        kill_tree(process)
        process.communicate()
        stderr = f"ERROR: timed out after {args.timeout} s"
    if process.returncode != 0 or not staged.exists():
        remove_staged(ytid)
        lines = [line for line in stderr.splitlines() if "ERROR" in line] or stderr.splitlines()
        return ytid, (lines[-1].strip() if lines else "no output file")[:300]

    seconds = sf.info(staged).duration
    if seconds < 9.0:
        staged.unlink()
        return ytid, f"clip too short ({seconds:.1f} s)"
    staged.replace(AUDIO / f"{ytid}.wav")
    return ytid, None


def load_failures() -> dict[str, tuple[str, bool]]:
    if not FAILED.exists():
        return {}
    with open(FAILED, newline="", encoding="utf-8") as f:
        failures = {}
        for row in csv.DictReader(f):
            # Re-check against the current patterns so wording added later also applies to old failures
            permanent = row["permanent"] == "True" or any(p in row["reason"].lower() for p in PERMANENT)
            failures[row["ytid"]] = (row["reason"], permanent)
        return failures


def save_failures(failures: dict[str, tuple[str, bool]]) -> None:
    with open(FAILED, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ytid", "reason", "permanent"])
        for ytid, (reason, permanent) in sorted(failures.items()):
            writer.writerow([ytid, reason, permanent])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="only consider the first N clips of the CSV")
    parser.add_argument(
        "--timeout", type=int, default=45,
        help="seconds before giving up on one clip; most take 3-5 s (default: 45)",
    )
    parser.add_argument("--js-runtime", default="node", help="node, deno or bun (default: node)")
    parser.add_argument(
        "--cookies-from-browser", metavar="BROWSER", help="use a browser's YouTube login if YouTube starts blocking"
    )
    parser.add_argument("--retry-failed", action="store_true", help="also retry videos that were unavailable before")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found on PATH")
    with open(MUSICCAPS / "musiccaps-public.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    failures = load_failures()
    todo = [
        row for row in rows
        if not (AUDIO / f"{row['ytid']}.wav").exists()
        and (args.retry_failed or not failures.get(row["ytid"], ("", False))[1])
    ]
    print(f"{len(rows) - len(todo)} of {len(rows)} clips already saved or known unavailable; fetching {len(todo)}", flush=True)
    shutil.rmtree(STAGING, ignore_errors=True)
    STAGING.mkdir(parents=True, exist_ok=True)

    saved = failed = blocked_streak = 0
    started = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download_clip, r["ytid"], int(r["start_s"]), int(r["end_s"]), args) for r in todo]
        for i, future in enumerate(as_completed(futures), 1):
            ytid, error = future.result()
            if error is None:
                saved += 1
                blocked_streak = 0
                failures.pop(ytid, None)
            else:
                failed += 1
                message = error.lower()
                failures[ytid] = (error, any(p in message for p in PERMANENT))
                blocked_streak = blocked_streak + 1 if any(b in message for b in BLOCKED) else 0
                if blocked_streak >= 25:
                    save_failures(failures)
                    pool.shutdown(cancel_futures=True)
                    sys.exit("YouTube is blocking requests. Wait a while, or re-run with --cookies-from-browser <browser>.")
            if i % 100 == 0 or i == len(todo):
                save_failures(failures)
                rate = i / (time.time() - started)
                eta = (len(todo) - i) / rate / 60
                print(f"{i}/{len(todo)} | saved {saved} | failed {failed} | {rate * 60:.0f} clips/min | ~{eta:.0f} min left", flush=True)

    permanent = sum(1 for _, is_permanent in failures.values() if is_permanent)
    print(f"Done. {len(list(AUDIO.glob('*.wav')))} clips on disk; {permanent} unavailable, {len(failures) - permanent} to retry")


if __name__ == "__main__":
    main()
