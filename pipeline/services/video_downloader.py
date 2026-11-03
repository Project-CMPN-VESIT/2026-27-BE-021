"""VideoDownloader service — yt-dlp wrapper for audio-only download and stream URL fetch.

Uses asyncio.get_event_loop().run_in_executor() rather than asyncio.create_subprocess_exec
so it works on BOTH Windows SelectorEventLoop (uvicorn default) and ProactorEventLoop.
"""
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _get_base_cmd() -> List[str]:
    """
    Get the base command to invoke yt-dlp with appropriate flags and JS runtime.
    Uses sys.executable -m yt_dlp to ensure packages in current python environment are used.
    """
    cmd = [sys.executable, "-m", "yt_dlp", "--no-update", "--no-playlist"]

    # Detect available JS runtime for YouTube signature deciphering (bun, deno, node)
    bun_path = shutil.which("bun")
    if not bun_path:
        local_bun = Path.home() / ".bun" / "bin" / "bun.exe"
        if local_bun.exists():
            bun_path = str(local_bun)

    if bun_path and os.path.exists(bun_path):
        cmd.extend(["--js-runtimes", f"bun:{bun_path}"])
    elif shutil.which("deno"):
        cmd.extend(["--js-runtimes", "deno"])
    elif shutil.which("node"):
        cmd.extend(["--js-runtimes", "node"])

    return cmd


def _run_subprocess(cmd: list) -> subprocess.CompletedProcess:
    """Run a subprocess synchronously (called inside a thread executor)."""
    return subprocess.run(cmd, capture_output=True, text=True)


async def get_stream_url(youtube_url: str) -> str:
    """
    Fetch the direct CDN video stream URL via yt-dlp (no download).
    Runs in a thread executor so it is compatible with any asyncio event loop.
    """
    cmd = _get_base_cmd() + [
        "--get-url",
        "-f", "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/best[ext=mp4]/best",
        youtube_url,
    ]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_subprocess,
        cmd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp get-url failed (code {result.returncode}): {result.stderr[:500]}"
        )
    url = result.stdout.strip().split("\n")[0]
    if not url:
        raise RuntimeError("yt-dlp returned empty stream URL")
    return url


async def download_audio(youtube_url: str, out_path: Path) -> Path:
    """
    Download only the audio track as MP3.
    Runs in a thread executor so it is compatible with any asyncio event loop.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _get_base_cmd() + [
        "-f", "bestaudio/best",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", str(out_path),
        youtube_url,
    ]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_subprocess,
        cmd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio download failed (code {result.returncode}): {result.stderr[:500]}"
        )
    if not out_path.exists():
        candidates = list(out_path.parent.glob("audio.*"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Audio file not found after download: {out_path}")
    return out_path


async def download_video(youtube_url: str, out_path: Path) -> Path:
    """
    Download a single video file that already includes audio.
    Runs in a thread executor so it is compatible with any asyncio event loop.
    
    Returns the path to the downloaded video file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _get_base_cmd() + [
        "-f", "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        youtube_url,
    ]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_subprocess,
        cmd,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Video download failed (code {result.returncode}): {result.stderr[:500]}"
        )
    if not out_path.exists():
        candidates = list(out_path.parent.glob("video.*"))
        if candidates:
            return candidates[0]
        # Check any downloaded media file in that folder
        all_candidates = [p for p in out_path.parent.glob("*.*") if p.suffix.lower() in [".mp4", ".mkv", ".webm"]]
        if all_candidates:
            return all_candidates[0]
        raise FileNotFoundError(f"Video file not found after download: {out_path}")
    return out_path


async def get_video_id(youtube_url: str) -> str:
    """Extract the YouTube video ID from a URL."""
    cmd = _get_base_cmd() + [
        "--get-id",
        youtube_url,
    ]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_subprocess,
        cmd,
    )
    return result.stdout.strip()

