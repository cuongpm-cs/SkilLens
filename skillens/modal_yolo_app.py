"""Modal service for GPU YOLO + CPU MediaPipe perception."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "skillens-yolo-perception"
VOLUME_NAME = "skillens-yolo-data"
VOLUME_ROOT = Path("/data")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy==1.26.4",
        "opencv-python==4.11.0.86",
        "mediapipe==0.10.21",
        "ultralytics==8.3.202",
        "lap>=0.5.12,<1",
    )
    .add_local_python_source("skillens")
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    timeout=7200,
    scaledown_window=600,
    volumes={str(VOLUME_ROOT): volume},
)
def run_yolo_perception(
    run_id: str,
    config: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Run one cacheable perception job from inputs stored on the Modal Volume."""
    from skillens.local_yolo import run_perception

    input_dir = VOLUME_ROOT / "inputs" / run_id
    output_dir = VOLUME_ROOT / "outputs" / run_id
    video = input_dir / "source.mp4"
    model = input_dir / "model.pt"
    perception_path = output_dir / "perception.json"
    if not video.is_file() or not model.is_file():
        raise FileNotFoundError(f"Missing Modal input files for run {run_id}")
    if perception_path.is_file() and not force:
        return json.loads(perception_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_perception(
        video,
        model,
        output_dir,
        confidence=float(config["confidence"]),
        iou=float(config["iou"]),
        device="0",
        tracker=str(config["tracker"]),
        event_gap_seconds=float(config.get("event_gap_seconds", 2.0)),
        event_padding_seconds=float(config.get("event_padding_seconds", 2.0)),
        max_event_seconds=float(config.get("max_event_seconds", 20.0)),
    )
    result["remote"] = {
        "app": APP_NAME,
        "volume": VOLUME_NAME,
        "run_id": run_id,
        "gpu": "L40S",
    }
    perception_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    volume.commit()
    return result


@app.function(image=image, volumes={str(VOLUME_ROOT): volume})
def health() -> dict[str, str]:
    """Small deployment check that does not allocate a GPU."""
    return {"status": "ok", "app": APP_NAME, "volume": VOLUME_NAME}

