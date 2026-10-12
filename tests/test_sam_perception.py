from pathlib import Path

from skillens.perception import _sam_run_id, _sanitize_sam_result


def test_sam_run_id_is_ascii_safe_for_non_latin_video_names(tmp_path: Path):
    video = tmp_path / "測試影片.mp4"
    manifest = tmp_path / "manifest.tsv"
    video.write_bytes(b"video")
    manifest.write_text("manifest", encoding="utf-8")

    run_id = _sam_run_id(video, 2.0, manifest)

    assert run_id.startswith("unified-video-")
    assert all(character.isascii() and (character.isalnum() or character in "-_") for character in run_id)


def test_sam_postprocessing_filters_confidence_and_limits_instances():
    result = {
        "class": "vial",
        "detections": [
            {"frame_index": 1, "probability": -10000.0, "bbox_xywh": [0, 0, 1, 1]},
            {"frame_index": 1, "probability": 0.8, "bbox_xywh": [0, 0, 1, 1]},
            {"frame_index": 1, "probability": 0.7, "bbox_xywh": [1, 1, 1, 1]},
            {"frame_index": 2, "probability": 0.2, "bbox_xywh": [0, 0, 1, 1]},
        ],
    }

    filtered = _sanitize_sam_result(result, threshold=0.35, max_instances=1)

    assert filtered["remote_detection_count"] == 4
    assert filtered["detection_count"] == 1
    assert filtered["detections"][0]["probability"] == 0.8
    assert filtered["local_postprocessing"]["removed_detections"] == 3
