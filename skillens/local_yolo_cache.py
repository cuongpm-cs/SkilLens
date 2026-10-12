"""Validated cache support for the local perception stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 1
PERCEPTION_ALGORITHM_VERSION = "hybrid-yolo-bytetrack-mediapipe-v1"
DEFAULT_ARTIFACTS = {
    "annotated_video": "video_with_boxes.mp4",
    "detections_csv": "detections.csv",
    "perception_json": "perception.json",
}


def _sampled_file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Fingerprint a large file without reading the entire file on every run."""
    path = path.expanduser().resolve()
    stat = path.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as file:
        if size <= chunk_size * 3:
            digest.update(file.read())
        else:
            for offset in (0, max(0, size // 2 - chunk_size // 2), size - chunk_size):
                file.seek(offset)
                digest.update(file.read(chunk_size))
    return {
        "name": path.name,
        "size_bytes": size,
        "sampled_sha256": digest.hexdigest(),
    }


def _tracker_fingerprint(tracker: str) -> dict[str, Any]:
    tracker_path = Path(tracker).expanduser()
    if tracker_path.is_file():
        return _sampled_file_fingerprint(tracker_path)
    return {"name": str(tracker), "builtin": True}


def build_perception_cache_manifest(
    *,
    video_path: Path,
    model_path: Path,
    tracker: str,
    confidence: float,
    iou: float,
    event_gap_seconds: float,
    event_padding_seconds: float,
    max_event_seconds: float,
) -> dict[str, Any]:
    """Build the Stage 1 identity; LLM settings are intentionally excluded."""
    inputs = {
        "source_video": _sampled_file_fingerprint(video_path),
        "yolo_model": _sampled_file_fingerprint(model_path),
        "tracker": _tracker_fingerprint(tracker),
        "parameters": {
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "hand_padding": 24,
            "max_num_hands": 2,
            "hand_detection_confidence": 0.5,
            "hand_tracking_confidence": 0.5,
            "event_gap_seconds": event_gap_seconds,
            "event_padding_seconds": event_padding_seconds,
            "max_event_seconds": max_event_seconds,
        },
    }
    key_material = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "algorithm_version": PERCEPTION_ALGORITHM_VERSION,
        "inputs": inputs,
    }
    key = hashlib.sha256(
        json.dumps(key_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**key_material, "key": key}


def _rebase_artifacts(perception: dict[str, Any], output_dir: Path) -> list[Path]:
    artifacts = perception.setdefault("artifacts", {})
    required: list[Path] = []
    for name, filename in DEFAULT_ARTIFACTS.items():
        local_path = (output_dir / filename).resolve()
        artifacts[name] = str(local_path)
        required.append(local_path)
    return required


def _legacy_cache_matches(
    perception: dict[str, Any], expected: dict[str, Any]
) -> tuple[bool, str]:
    """Conservatively recognize outputs written before cache manifests existed."""
    metadata = perception.get("metadata", {})
    inputs = expected["inputs"]
    parameters = inputs["parameters"]
    comparisons = (
        (Path(str(metadata.get("source_video", ""))).name, inputs["source_video"]["name"]),
        (Path(str(metadata.get("model", ""))).name, inputs["yolo_model"]["name"]),
        (Path(str(metadata.get("tracker", ""))).name, inputs["tracker"]["name"]),
        (metadata.get("confidence_threshold"), parameters["confidence_threshold"]),
        (metadata.get("iou_threshold"), parameters["iou_threshold"]),
    )
    if not all(actual == wanted for actual, wanted in comparisons):
        return False, "legacy metadata does not match the current video or detector"
    legacy_event_defaults = (
        parameters["event_gap_seconds"] == 2.0
        and parameters["event_padding_seconds"] == 2.0
        and parameters["max_event_seconds"] == 20.0
    )
    if not legacy_event_defaults:
        return False, "legacy output cannot validate non-default event settings"
    return True, "legacy Stage 1 output migrated to the validated cache format"


def load_perception_cache(
    *,
    perception_path: Path,
    output_dir: Path,
    expected_manifest: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Return a valid cached result and a human-readable cache decision."""
    if not perception_path.is_file():
        return None, "perception.json does not exist"
    try:
        perception = json.loads(perception_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"perception.json is unreadable: {error}"
    if perception.get("metadata", {}).get("timeline_coverage") != "hybrid_full_video":
        return None, "cached events do not cover the complete video timeline"
    if not isinstance(perception.get("events"), list):
        return None, "cached events are missing"

    stored_manifest = perception.get("cache")
    migrated = False
    if stored_manifest is None:
        matches, reason = _legacy_cache_matches(perception, expected_manifest)
        if not matches:
            return None, reason
        migrated = True
    elif stored_manifest.get("key") != expected_manifest["key"]:
        return None, "video, YOLO model, tracker, or perception settings changed"

    missing = [path for path in _rebase_artifacts(perception, output_dir) if not path.is_file()]
    if missing:
        return None, "cached artifact is missing: " + ", ".join(path.name for path in missing)

    perception["cache"] = expected_manifest
    if migrated:
        write_perception_cache(perception, perception_path, expected_manifest, output_dir)
        return perception, reason
    return perception, "cache key and artifacts are valid"


def write_perception_cache(
    perception: dict[str, Any],
    perception_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
) -> None:
    """Attach cache provenance and atomically persist perception.json."""
    perception["cache"] = manifest
    _rebase_artifacts(perception, output_dir)
    temporary = perception_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(perception, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(perception_path)
