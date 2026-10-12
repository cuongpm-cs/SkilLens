"""Local perception stage: YOLO tools, hand landmarks, tracking and events."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from ultralytics import YOLO


def _hand_boxes(result: Any, width: int, height: int, padding: int) -> list[list[int]]:
    boxes: list[list[int]] = []
    if not result.multi_hand_landmarks:
        return boxes
    for landmarks in result.multi_hand_landmarks:
        xs = [int(point.x * width) for point in landmarks.landmark]
        ys = [int(point.y * height) for point in landmarks.landmark]
        boxes.append(
            [
                max(0, min(xs) - padding),
                max(0, min(ys) - padding),
                min(width - 1, max(xs) + padding),
                min(height - 1, max(ys) + padding),
            ]
        )
    return boxes


def _boxes_overlap(first: list[float], second: list[float]) -> bool:
    return not (
        first[0] > second[2]
        or second[0] > first[2]
        or first[1] > second[3]
        or second[1] > first[3]
    )


def _contact_hand(tool_box: list[float], hands: list[list[int]]) -> int | None:
    for index, hand_box in enumerate(hands):
        if _boxes_overlap(tool_box, hand_box):
            return index
    return None


def _make_events(
    detections: list[dict[str, Any]],
    *,
    fps: float,
    duration: float,
    gap_seconds: float,
    padding_seconds: float,
    max_event_seconds: float,
) -> list[dict[str, Any]]:
    contact_frames = sorted(
        {row["frame_index"] for row in detections if row["hand_contact"]}
    )
    event_basis = "hand_tool_contact"
    if not contact_frames:
        contact_frames = sorted({row["frame_index"] for row in detections})
        event_basis = "tool_detection_fallback"
    if not contact_frames:
        return []

    max_gap_frames = max(1, round(gap_seconds * fps))
    groups: list[list[int]] = [[contact_frames[0]]]
    for frame_index in contact_frames[1:]:
        if frame_index - groups[-1][-1] <= max_gap_frames:
            groups[-1].append(frame_index)
        else:
            groups.append([frame_index])

    intervals: list[tuple[float, float, str]] = []
    for group in groups:
        start = max(0.0, group[0] / fps - padding_seconds)
        end = min(duration, group[-1] / fps + padding_seconds)
        while end - start > max_event_seconds:
            intervals.append((start, start + max_event_seconds, event_basis))
            start += max_event_seconds
        intervals.append((start, end, event_basis))

    # Tool-centered events are strong candidates, but many rubrics also contain
    # long-duration or no-tool actions. Add uniform windows for every uncovered
    # part of the video so those actions remain observable.
    coverage_intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, _ in sorted(intervals):
        gap_end = min(duration, start)
        while cursor + 0.001 < gap_end:
            window_end = min(gap_end, cursor + max_event_seconds)
            coverage_intervals.append(
                (cursor, window_end, "uniform_timeline_coverage")
            )
            cursor = window_end
        cursor = max(cursor, end)
    while cursor + 0.001 < duration:
        window_end = min(duration, cursor + max_event_seconds)
        coverage_intervals.append((cursor, window_end, "uniform_timeline_coverage"))
        cursor = window_end
    intervals.extend(coverage_intervals)
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))

    events: list[dict[str, Any]] = []
    for event_index, (start, end, basis) in enumerate(intervals):
        rows = [row for row in detections if start <= row["time_seconds"] <= end]
        by_class: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = by_class.setdefault(
                row["class_name"],
                {
                    "class_id": row["class_id"],
                    "detection_count": 0,
                    "contact_count": 0,
                    "max_confidence": 0.0,
                    "track_ids": set(),
                },
            )
            item["detection_count"] += 1
            item["contact_count"] += int(row["hand_contact"])
            item["max_confidence"] = max(item["max_confidence"], row["confidence"])
            if row["track_id"] is not None:
                item["track_ids"].add(row["track_id"])

        objects = []
        for name, item in sorted(by_class.items()):
            objects.append(
                {
                    "class_name": name,
                    "class_id": item["class_id"],
                    "detection_count": item["detection_count"],
                    "contact_count": item["contact_count"],
                    "max_confidence": round(item["max_confidence"], 6),
                    "track_ids": sorted(item["track_ids"]),
                }
            )
        events.append(
            {
                "event_id": f"event_{event_index:03d}",
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "basis": basis,
                "objects": objects,
            }
        )
    return events


def run_perception(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    confidence: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
    hand_padding: int = 24,
    event_gap_seconds: float = 2.0,
    event_padding_seconds: float = 2.0,
    max_event_seconds: float = 20.0,
) -> dict[str, Any]:
    """Analyze a video locally and return paths plus structured candidate events."""
    video_path = video_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = output_dir / "video_with_boxes.mp4"
    csv_path = output_dir / "detections.csv"
    evidence_path = output_dir / "perception.json"

    model = YOLO(str(model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {annotated_path}")

    predict_options: dict[str, Any] = {
        "persist": True,
        "tracker": tracker,
        "conf": confidence,
        "iou": iou,
        "verbose": False,
    }
    if device is not None:
        predict_options["device"] = device

    rows: list[dict[str, Any]] = []
    frame_index = 0
    hands_api = mp.solutions.hands
    try:
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                hand_result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_boxes = _hand_boxes(hand_result, width, height, hand_padding)
                result = model.track(frame, **predict_options)[0]
                annotated = result.plot()
                for hand_index, (x1, y1, x2, y2) in enumerate(hand_boxes):
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        annotated,
                        f"hand {hand_index}",
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(annotated)

                if result.boxes is not None and len(result.boxes):
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    scores = result.boxes.conf.cpu().numpy()
                    track_ids: list[int | None]
                    if result.boxes.id is None:
                        track_ids = [None] * len(xyxy)
                    else:
                        track_ids = [int(value) for value in result.boxes.id.cpu().numpy()]
                    for box, class_id, score, track_id in zip(
                        xyxy, class_ids, scores, track_ids
                    ):
                        pixel_box = [float(value) for value in box.tolist()]
                        hand_index = _contact_hand(pixel_box, hand_boxes)
                        rows.append(
                            {
                                "frame_index": frame_index,
                                "time_seconds": round(frame_index / fps, 3),
                                "class_id": int(class_id),
                                "class_name": str(model.names[int(class_id)]),
                                "track_id": track_id,
                                "confidence": round(float(score), 6),
                                "bbox_xyxy": [round(value, 2) for value in pixel_box],
                                "bbox_normalized": [
                                    round(pixel_box[0] / width, 6),
                                    round(pixel_box[1] / height, 6),
                                    round(pixel_box[2] / width, 6),
                                    round(pixel_box[3] / height, 6),
                                ],
                                "hand_contact": hand_index is not None,
                                "hand_index": hand_index,
                            }
                        )

                frame_index += 1
                if frame_index % 100 == 0:
                    suffix = f"/{total_frames}" if total_frames > 0 else ""
                    print(f"Perception: {frame_index}{suffix} frames")
    finally:
        capture.release()
        writer.release()

    if duration <= 0:
        duration = frame_index / fps
    events = _make_events(
        rows,
        fps=fps,
        duration=duration,
        gap_seconds=event_gap_seconds,
        padding_seconds=event_padding_seconds,
        max_event_seconds=max_event_seconds,
    )
    if not events:
        # YOLO failure must not completely prevent VLM review.
        start = 0.0
        while start < duration:
            end = min(duration, start + max_event_seconds)
            events.append(
                {
                    "event_id": f"event_{len(events):03d}",
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                    "basis": "uniform_fallback_no_detections",
                    "objects": [],
                }
            )
            start = end

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.writer(file)
        writer_csv.writerow(
            [
                "frame_index",
                "time_seconds",
                "class_id",
                "class_name",
                "track_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "hand_contact",
                "hand_index",
            ]
        )
        for row in rows:
            writer_csv.writerow(
                [
                    row["frame_index"],
                    row["time_seconds"],
                    row["class_id"],
                    row["class_name"],
                    row["track_id"],
                    row["confidence"],
                    *row["bbox_xyxy"],
                    row["hand_contact"],
                    row["hand_index"],
                ]
            )

    track_summary: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"frames": 0, "contacts": 0, "max_confidence": 0.0}
    )
    for row in rows:
        key = f"{row['class_name']}:{row['track_id']}"
        track_summary[key]["frames"] += 1
        track_summary[key]["contacts"] += int(row["hand_contact"])
        track_summary[key]["max_confidence"] = max(
            track_summary[key]["max_confidence"], row["confidence"]
        )
    for summary in track_summary.values():
        summary["max_confidence"] = round(summary["max_confidence"], 6)

    evidence = {
        "metadata": {
            "source_video": str(video_path),
            "model": str(model_path),
            "fps": fps,
            "width": width,
            "height": height,
            "frames": frame_index,
            "duration_seconds": round(duration, 3),
            "classes": {str(key): value for key, value in model.names.items()},
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "tracker": tracker,
            "timeline_coverage": "hybrid_full_video",
        },
        "tracks": dict(track_summary),
        "events": events,
        "detections_count": len(rows),
        "contacts_count": sum(int(row["hand_contact"]) for row in rows),
        "artifacts": {
            "annotated_video": str(annotated_path),
            "detections_csv": str(csv_path),
            "perception_json": str(evidence_path),
        },
    }
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evidence
