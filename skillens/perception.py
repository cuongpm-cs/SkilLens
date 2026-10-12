"""Perception adapters for YOLO, SAM 3.1, and API-only operation."""

from __future__ import annotations

import hashlib
import json
import csv
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import local_yolo_cache
from .media import probe_video, uniform_events


def run_full_api(video: Path, output_dir: Path, *, event_seconds: float) -> dict[str, Any]:
    """Create full-timeline evidence windows without a detector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(video)
    perception = {
        "schema_version": "1.0",
        "backend": "full_api",
        "metadata": {**metadata, "timeline_coverage": "uniform_full_video"},
        "events": uniform_events(metadata["duration"], event_seconds),
        "artifacts": {"perception_json": str(output_dir / "perception.json")},
    }
    path = output_dir / "perception.json"
    path.write_text(json.dumps(perception, ensure_ascii=False, indent=2), encoding="utf-8")
    return perception


def run_yolo(
    video: Path,
    model: Path,
    output_dir: Path,
    *,
    confidence: float,
    iou: float,
    device: str | None,
    tracker: str,
    force: bool,
) -> dict[str, Any]:
    """Call the authoritative YOLO + ByteTrack + MediaPipe implementation."""
    from . import local_yolo

    model = model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = local_yolo_cache.build_perception_cache_manifest(
        video_path=video,
        model_path=model,
        tracker=tracker,
        confidence=confidence,
        iou=iou,
        event_gap_seconds=2.0,
        event_padding_seconds=2.0,
        max_event_seconds=20.0,
    )
    perception_path = output_dir / "perception.json"
    if not force:
        result, reason = local_yolo_cache.load_perception_cache(
            perception_path=perception_path,
            output_dir=output_dir,
            expected_manifest=manifest,
        )
        if result is not None:
            print(f"Perception cache hit: {reason}")
            result["backend"] = "yolo"
            return result
        print(f"Perception cache miss: {reason}")
    result = local_yolo.run_perception(
        video,
        model,
        output_dir,
        confidence=confidence,
        iou=iou,
        device=device,
        tracker=tracker,
    )
    result["backend"] = "yolo"
    local_yolo_cache.write_perception_cache(result, perception_path, manifest, output_dir)
    return result


def run_yolo_modal(
    video: Path,
    model: Path,
    output_dir: Path,
    *,
    confidence: float,
    iou: float,
    tracker: str,
    force: bool,
    modal_app: str,
    modal_volume: str,
) -> dict[str, Any]:
    """Call the deployed Modal YOLO + MediaPipe service."""
    from .modal_yolo import run_modal_perception

    return run_modal_perception(
        video,
        model,
        output_dir,
        confidence=confidence,
        iou=iou,
        tracker=tracker,
        force=force,
        app_name=modal_app,
        volume_name=modal_volume,
    )


def _sam_run_id(video: Path, analysis_fps: float, manifest: Path) -> str:
    stat = video.stat()
    material = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{analysis_fps}:{manifest.resolve()}"
    ascii_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", video.stem).strip("-_")
    safe_stem = ascii_stem[:24] or "video"
    return f"unified-{safe_stem}-{hashlib.sha256(material.encode()).hexdigest()[:12]}"


def _read_sam_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"class", "source_frame", "x", "y", "width", "height"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"SAM manifest must contain columns: {sorted(required)}")
    normalized = []
    for row in rows:
        item = dict(row)
        for key in ("source_frame", "x", "y", "width", "height"):
            item[key] = int(float(item[key]))
        normalized.append(item)
    return normalized


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(4 * 1024 * 1024))
        if stat.st_size > 4 * 1024 * 1024:
            handle.seek(max(0, stat.st_size - 4 * 1024 * 1024))
            digest.update(handle.read())
    return {"name": path.name, "size": stat.st_size, "sampled_sha256": digest.hexdigest()}


def _upload_sam_inputs(volume: Any, run_id: str, video: Path, rubric: Path, manifest: Path) -> None:
    remote_dir = f"/inputs/{run_id}"
    payload = {
        "schema_version": "1.0",
        "video": _file_identity(video),
        "rubric": _file_identity(rubric),
        "manifest": _file_identity(manifest),
    }
    remote_manifest = f"{remote_dir}/input_manifest.json"
    try:
        current = json.loads(b"".join(volume.read_file(remote_manifest)).decode("utf-8"))
    except Exception:
        current = None
    if current == payload:
        print(f"SAM upload cache hit: {remote_dir}")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        local_manifest = Path(handle.name)
    try:
        with volume.batch_upload(force=True) as batch:
            if not current or current.get("video") != payload["video"]:
                batch.put_file(video, f"{remote_dir}/source.mp4")
            batch.put_file(rubric, f"{remote_dir}/rubric.json")
            batch.put_file(manifest, f"{remote_dir}/manifest.tsv")
            batch.put_file(local_manifest, remote_manifest)
    finally:
        local_manifest.unlink(missing_ok=True)


def _sam_events(
    duration: float,
    event_seconds: float,
    sam_results: list[dict[str, Any]],
    graph: dict[str, Any],
) -> list[dict[str, Any]]:
    events = uniform_events(duration, event_seconds)
    detections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in sam_results:
        detections[str(result["class"])].extend(result.get("detections", []))
    contacts = graph.get("frame_edges", [])
    for event in events:
        start, end = event["start_seconds"], event["end_seconds"]
        objects = []
        for class_name, rows in detections.items():
            selected = [row for row in rows if start <= float(row["timestamp"]) <= end]
            if not selected:
                continue
            contact_count = sum(
                start <= float(edge["timestamp"]) <= end
                and edge.get("tool_class") == class_name
                for edge in contacts
            )
            probabilities = [
                float(row["probability"])
                for row in selected
                if row.get("probability") is not None
            ]
            objects.append(
                {
                    "class_name": class_name,
                    "detection_count": len(selected),
                    "contact_count": contact_count,
                    "max_confidence": max(probabilities) if probabilities else None,
                    "track_ids": [],
                }
            )
        event["objects"] = objects
        event["basis"] = "sam31_mediapipe_full_timeline"
    return events


def _sanitize_sam_result(
    result: dict[str, Any],
    *,
    threshold: float,
    max_instances: int | None,
) -> dict[str, Any]:
    """Remove invalid propagation outputs and bound same-class instances."""
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_detections = list(result.get("detections") or [])
    for detection in raw_detections:
        probability = detection.get("probability")
        try:
            confidence = float(probability)
        except (TypeError, ValueError):
            continue
        if confidence < threshold:
            continue
        by_frame[int(detection["frame_index"])].append(detection)

    detections: list[dict[str, Any]] = []
    for frame_detections in by_frame.values():
        frame_detections.sort(
            key=lambda item: float(item.get("probability", -1.0)), reverse=True
        )
        if max_instances is not None:
            frame_detections = frame_detections[:max_instances]
        detections.extend(frame_detections)
    detections.sort(
        key=lambda item: (
            int(item["frame_index"]),
            float(item["bbox_xywh"][0]),
            float(item["bbox_xywh"][1]),
        )
    )
    return {
        **result,
        "remote_detection_count": len(raw_detections),
        "detection_count": len(detections),
        "detections": detections,
        "local_postprocessing": {
            "confidence_threshold": threshold,
            "max_instances_per_frame": max_instances,
            "removed_detections": len(raw_detections) - len(detections),
        },
    }


def run_sam31(
    video: Path,
    rubric_path: Path,
    manifest: Path,
    output_dir: Path,
    *,
    analysis_fps: float,
    event_seconds: float,
    threshold: float,
    max_anchors: int,
    hand_chunk_frames: int,
    force: bool,
    modal_app: str,
    modal_volume: str,
    class_prompts: dict[str, str] | None = None,
    class_max_instances: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Reuse the official Modal SAM 3.1 and MediaPipe perception stages."""
    try:
        import modal
    except ImportError as error:
        raise RuntimeError("SAM 3.1 backend requires the 'modal' package") from error
    manifest = manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"SAM exemplar manifest not found: {manifest}")
    rows = _read_sam_manifest(manifest)
    anchors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        anchors[row["class"]].append(row)
    if not anchors:
        raise ValueError(f"SAM manifest contains no classes: {manifest}")
    run_id = _sam_run_id(video, analysis_fps, manifest)
    volume = modal.Volume.from_name(modal_volume, create_if_missing=True)
    _upload_sam_inputs(volume, run_id, video, rubric_path, manifest)
    prepare = modal.Function.from_name(modal_app, "prepare_video")
    segment = modal.Function.from_name(modal_app, "segment_class_v5")
    hands = modal.Function.from_name(modal_app, "detect_hands_chunk")
    graph_function = modal.Function.from_name(modal_app, "build_graph")
    metadata = prepare.remote(run_id, analysis_fps, None, force)
    print(f"SAM remote preparation: {metadata['frame_count']} frames")
    calls = {
        class_name: segment.spawn(
            run_id,
            class_name,
            (class_prompts or {}).get(class_name, class_name.replace("_", " ")),
            class_anchors,
            threshold,
            3.0,
            max_anchors,
            force,
        )
        for class_name, class_anchors in anchors.items()
    }
    chunk = max(25, hand_chunk_frames)
    hand_calls = [
        hands.spawn(run_id, start, min(metadata["frame_count"], start + chunk), force)
        for start in range(0, metadata["frame_count"], chunk)
    ]
    sam_results = []
    for class_name, call in calls.items():
        raw_result = call.get()
        configured_limit = (class_max_instances or {}).get(class_name)
        max_instances = int(configured_limit) if configured_limit is not None else None
        if max_instances is not None and max_instances <= 0:
            raise ValueError(f"SAM max instances must be positive for {class_name}")
        sam_results.append(
            _sanitize_sam_result(
                raw_result,
                threshold=threshold,
                max_instances=max_instances,
            )
        )
    hand_results = [call.get() for call in hand_calls]
    graph = graph_function.remote(run_id, sam_results, hand_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    perception = {
        "schema_version": "1.0",
        "backend": "sam31",
        "metadata": {
            **metadata,
            "source_video": str(video),
            "timeline_coverage": "uniform_full_video",
        },
        "sam": sam_results,
        "hands": hand_results,
        "graph_summary": graph.get("counts", {}),
        "stage_config": {
            "manifest": str(manifest),
            "manifest_size": manifest.stat().st_size,
            "manifest_mtime_ns": manifest.stat().st_mtime_ns,
            "analysis_fps": analysis_fps,
            "event_seconds": event_seconds,
            "threshold": threshold,
            "max_anchors": max_anchors,
            "hand_chunk_frames": hand_chunk_frames,
            "modal_app": modal_app,
            "modal_volume": modal_volume,
            "max_instances": class_max_instances or {},
        },
        "events": _sam_events(metadata["duration"], event_seconds, sam_results, graph),
        "artifacts": {
            "graph_json": str(graph_path),
            "perception_json": str(output_dir / "perception.json"),
        },
    }
    perception_path = output_dir / "perception.json"
    perception_path.write_text(
        json.dumps(perception, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return perception
