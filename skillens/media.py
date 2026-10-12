"""Video probing and timeline utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def probe_video(video: Path) -> dict[str, Any]:
    video = video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=index,codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required but was not found on PATH") from error
    payload = json.loads(result.stdout)
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found: {video}")
    return {
        "source_video": str(video),
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "source_fps": video_stream.get("r_frame_rate"),
        "audio_available": any(
            stream.get("codec_type") == "audio" for stream in payload.get("streams", [])
        ),
    }


def uniform_events(duration: float, seconds: float = 30.0) -> list[dict[str, Any]]:
    if seconds <= 0:
        raise ValueError("Timeline window must be positive")
    if duration <= 0:
        raise ValueError("Video duration must be positive")
    events = []
    start = 0.0
    while start < duration:
        end = min(duration, start + seconds)
        events.append(
            {
                "event_id": f"event_{len(events):03d}",
                "start_seconds": round(start, 3),
                "end_seconds": duration if end == duration else round(end, 3),
                "basis": "full_timeline_coverage",
                "objects": [],
            }
        )
        start = end
    return events
