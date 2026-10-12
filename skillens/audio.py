"""Optional cached speech evidence for criteria that require audio."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from .media import probe_video
from .rubric import rubric_requires_audio


def should_use_audio(
    mode: str,
    *,
    rubric: dict[str, Any],
    profile_requires_audio: bool,
    video_has_audio: bool,
) -> tuple[bool, str]:
    if mode == "never":
        return False, "disabled by --audio never"
    needed = profile_requires_audio or rubric_requires_audio(rubric)
    if mode == "auto" and not needed:
        return False, "the selected rubric has no audio-dependent criterion"
    if not video_has_audio:
        return False, "the video has no audio stream"
    return True, "explicitly enabled" if mode == "always" else "required by exam rubric"


def _extract_chunk(video: Path, start: float, duration: float, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "48k",
            str(destination),
        ],
        check=True,
    )


def build_audio_evidence(
    video: Path,
    output_dir: Path,
    *,
    enabled: bool,
    activation_reason: str,
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    language: str | None,
    chunk_seconds: float,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "audio_evidence.json"
    if not enabled:
        result = {
            "schema_version": "1.0",
            "enabled": False,
            "activation_reason": activation_reason,
            "chunks": [],
        }
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    if chunk_seconds <= 0:
        raise ValueError("Audio chunk length must be positive")
    if provider not in {"openai", "openai-compatible"}:
        raise ValueError("Audio provider must be openai or openai-compatible")
    api_key = api_key or os.getenv("AUDIO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Audio transcription requires AUDIO_API_KEY or OPENAI_API_KEY")
    options: dict[str, Any] = {"api_key": api_key, "timeout": 900.0}
    if base_url:
        options["base_url"] = base_url
    metadata = probe_video(video)
    identity = {
        "video": str(video.resolve()),
        "size": video.stat().st_size,
        "mtime_ns": video.stat().st_mtime_ns,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "language": language,
        "chunk_seconds": chunk_seconds,
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if path.is_file() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            print("Audio cache hit")
            return cached
    client = OpenAI(**options)
    windows = [
        (index, start, min(chunk_seconds, metadata["duration"] - start))
        for index, start in enumerate(
            index * chunk_seconds
            for index in range(math.ceil(metadata["duration"] / chunk_seconds))
        )
    ]

    def transcribe(item: tuple[int, float, float]) -> dict[str, Any]:
        index, start, duration = item
        with tempfile.TemporaryDirectory(prefix="skillens-audio-") as temporary:
            audio_path = Path(temporary) / "chunk.mp3"
            _extract_chunk(video, start, duration, audio_path)
            with audio_path.open("rb") as handle:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "file": handle,
                    "response_format": "text",
                }
                if language:
                    kwargs["language"] = language
                response = client.audio.transcriptions.create(**kwargs)
        text = response if isinstance(response, str) else getattr(response, "text", str(response))
        text = str(text).strip()
        return {
            "audio_id": f"audio_{index:04d}",
            "start_seconds": round(start, 3),
            "end_seconds": round(start + duration, 3),
            "text": text,
            "usable": bool(text),
            "speaker_verified": False,
        }

    chunks = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as executor:
        futures = [executor.submit(transcribe, item) for item in windows]
        for future in as_completed(futures):
            chunks.append(future.result())
    chunks.sort(key=lambda item: item["start_seconds"])
    result = {
        "schema_version": "1.0",
        "enabled": True,
        "activation_reason": activation_reason,
        "cache_key": cache_key,
        "provider": provider,
        "model": model,
        "speaker_policy": "speaker identity is unverified; use transcript as supporting evidence only",
        "chunks": chunks,
        "summary": {
            "total_chunks": len(chunks),
            "usable_chunks": sum(item["usable"] for item in chunks),
        },
    }
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
