"""Local client for the Modal YOLO + MediaPipe perception service."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import modal

from . import local_yolo_cache


DEFAULT_APP = "skillens-yolo-perception"
DEFAULT_VOLUME = "skillens-yolo-data"


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(4 * 1024 * 1024))
        if stat.st_size > 4 * 1024 * 1024:
            handle.seek(max(0, stat.st_size - 4 * 1024 * 1024))
            digest.update(handle.read())
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sampled_sha256": digest.hexdigest(),
    }


def _run_id(video: Path, model: Path, config: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "video": _file_identity(video),
            "model": _file_identity(model),
            "config": config,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"{video.stem[:24]}-{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _upload_inputs(
    volume: modal.Volume,
    *,
    run_id: str,
    video: Path,
    model: Path,
    payload: dict[str, Any],
) -> bool:
    remote_dir = f"/inputs/{run_id}"
    remote_manifest = f"{remote_dir}/input_manifest.json"
    try:
        current = json.loads(
            b"".join(volume.read_file(remote_manifest)).decode("utf-8")
        )
    except Exception:
        current = None
    if current == payload:
        print(f"Modal YOLO upload cache hit: {remote_dir}")
        return True

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        local_manifest = Path(handle.name)
    try:
        print(f"Uploading video and YOLO checkpoint to Modal Volume: {remote_dir}")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(video, f"{remote_dir}/source.mp4")
            batch.put_file(model, f"{remote_dir}/model.pt")
            batch.put_file(local_manifest, remote_manifest)
    finally:
        local_manifest.unlink(missing_ok=True)
    return False


def _download_file(volume: modal.Volume, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as handle:
        volume.read_file_into_fileobj(remote, handle)


def run_modal_perception(
    video: Path,
    model: Path,
    output_dir: Path,
    *,
    confidence: float,
    iou: float,
    tracker: str,
    force: bool,
    app_name: str = DEFAULT_APP,
    volume_name: str = DEFAULT_VOLUME,
) -> dict[str, Any]:
    """Upload inputs, call the deployed Modal function, and localize artifacts."""
    video = video.expanduser().resolve()
    model = model.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "confidence": confidence,
        "iou": iou,
        "tracker": tracker,
        "event_gap_seconds": 2.0,
        "event_padding_seconds": 2.0,
        "max_event_seconds": 20.0,
    }
    cache_manifest = local_yolo_cache.build_perception_cache_manifest(
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
        cached, reason = local_yolo_cache.load_perception_cache(
            perception_path=perception_path,
            output_dir=output_dir,
            expected_manifest=cache_manifest,
        )
        if cached is not None:
            print(f"Modal YOLO local cache hit: {reason}")
            cached["backend"] = "yolo"
            return cached
        print(f"Modal YOLO local cache miss: {reason}")

    run_id = _run_id(video, model, config)
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    input_payload = {
        "schema_version": "1.0",
        "video": _file_identity(video),
        "model": _file_identity(model),
        "config": config,
    }
    _upload_inputs(
        volume,
        run_id=run_id,
        video=video,
        model=model,
        payload=input_payload,
    )
    remote = modal.Function.from_name(app_name, "run_yolo_perception")
    result = remote.remote(run_id, config, force)

    remote_output = f"/outputs/{run_id}"
    local_artifacts = {
        "annotated_video": output_dir / "video_with_boxes.mp4",
        "detections_csv": output_dir / "detections.csv",
    }
    for key, local_path in local_artifacts.items():
        print(f"Downloading Modal artifact: {local_path.name}")
        _download_file(volume, f"{remote_output}/{local_path.name}", local_path)

    result["metadata"]["source_video"] = str(video)
    result["metadata"]["model"] = str(model)
    result["metadata"]["execution"] = "modal"
    result["artifacts"] = {
        **result.get("artifacts", {}),
        "annotated_video": str(local_artifacts["annotated_video"]),
        "detections_csv": str(local_artifacts["detections_csv"]),
        "perception_json": str(perception_path),
    }
    result["backend"] = "yolo"
    result["remote"] = {
        **result.get("remote", {}),
        "app": app_name,
        "volume": volume_name,
        "run_id": run_id,
    }
    local_yolo_cache.write_perception_cache(
        result, perception_path, cache_manifest, output_dir
    )
    return result
