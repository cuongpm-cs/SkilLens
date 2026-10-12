"""Local bounding-box annotation tool for catheterization OSCE frames.

The tool intentionally keeps the established video-level train/dev/test split
from ``osce-yolo-auto``.  Manual work is written to a separate project, while
the source images and silver labels remain read-only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import unicodedata
import uuid
import webbrowser
import zipfile
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


LEGACY_SUGGESTION_CLASS_NAMES = (
    "urinary_catheter",
    "10_ml_syringe",
    "catheterization_kit",
    "distilled_water_bottle",
    "sterile_gloves",
    "povidone_iodine_bottle",
)
CLASS_COLOR_PALETTE = (
    "#007C91",
    "#E57A44",
    "#6C5CE7",
    "#2D8BCB",
    "#00A878",
    "#C44E8B",
    "#946B2D",
    "#607D8B",
    "#D1495B",
    "#3A7D44",
)
VALID_SPLITS = ("train", "dev", "test")
VALID_STATUSES = {"unreviewed", "annotated", "empty", "needs_review"}
VIDEO_ID_RE = re.compile(r"video_[0-9a-f]{16}$")
FRAME_RE = re.compile(r"video_[0-9a-f]{16}_frame_(\d{6})\.jpg$")
BOX_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}$")
COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}$")

DEFAULT_FRAMES_RELATIVE = Path("data/catheterization/osce-yolo-auto/images")
DEFAULT_SPLIT_RELATIVE = Path("data/catheterization/osce-yolo-auto/images")
DEFAULT_SUGGESTIONS_RELATIVE = Path(
    "data/catheterization/osce-yolo-full6-silver-v4/labels"
)
DEFAULT_OUTPUT_RELATIVE = Path("data/catheterization/manual-annotation-3video")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_project_path(raw: str, project_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def manifest_class_names(manifest: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row["name"]) for row in manifest.get("classes", []))


def suggestions_are_compatible(manifest: dict[str, Any]) -> bool:
    """Legacy proposals are safe only when their exact original taxonomy is active."""
    return manifest_class_names(manifest) == LEGACY_SUGGESTION_CLASS_NAMES


def normalize_class_name(label: str, fallback_id: int) -> str:
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    name = re.sub(r"[^a-z0-9]+", "_", ascii_label.lower()).strip("_")
    if not name:
        name = f"class_{fallback_id}"
    if name[0].isdigit():
        name = f"object_{name}"
    return name[:80]


def build_class(label: str, color: str, existing: list[dict[str, Any]]) -> dict[str, Any]:
    label = " ".join(label.strip().split())
    if not label or len(label) > 80:
        raise ValueError("Class name must contain 1–80 characters.")
    if not COLOR_RE.fullmatch(color):
        raise ValueError("Class color must use #RRGGBB format.")
    class_id = len(existing)
    name = normalize_class_name(label, class_id)
    existing_names = {str(row["name"]).casefold() for row in existing}
    existing_labels = {str(row["label"]).casefold() for row in existing}
    if name.casefold() in existing_names or label.casefold() in existing_labels:
        raise ValueError("That class already exists.")
    return {"id": class_id, "name": name, "label": label, "color": color.upper()}


def count_suggestion_classes(label_dir: Path) -> list[int]:
    counts = [0] * len(LEGACY_SUGGESTION_CLASS_NAMES)
    if not label_dir.is_dir():
        return counts
    for path in label_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if 0 <= class_id < len(counts):
                counts[class_id] += 1
    return counts


def choose_representative_videos(
    frames_root: Path, suggestions_root: Path
) -> list[str]:
    """Pick one label-rich video per established split, deterministically."""
    selected: list[str] = []
    for split in VALID_SPLITS:
        split_root = frames_root / split
        candidates: list[tuple[tuple[int, int, int, str], str]] = []
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing frame split: {split_root}")
        for directory in sorted(split_root.iterdir()):
            if not directory.is_dir() or not VIDEO_ID_RE.fullmatch(directory.name):
                continue
            counts = count_suggestion_classes(suggestions_root / split / directory.name)
            coverage = sum(value > 0 for value in counts)
            minimum = min(counts) if coverage == len(LEGACY_SUGGESTION_CLASS_NAMES) else 0
            total = sum(counts)
            # Lexicographically smaller video IDs win an exact numerical tie.
            score = (coverage, minimum, total, "".join(chr(255 - ord(c)) for c in directory.name))
            candidates.append((score, directory.name))
        if not candidates:
            raise FileNotFoundError(f"No video directories in {split_root}")
        selected.append(max(candidates)[1])
    return selected


def discover_video(frames_root: Path, video_id: str) -> tuple[str, Path]:
    matches = [
        (split, frames_root / split / video_id)
        for split in VALID_SPLITS
        if (frames_root / split / video_id).is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{video_id} must occur in exactly one established split; found {len(matches)}."
        )
    return matches[0]


def link_image(source: Path, destination: Path) -> str:
    """Materialize a zero-copy training view when the filesystem permits it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            # The annotation UI can still serve the source image directly.
            return "source_only"


def parse_frame_index(filename: str) -> int:
    match = FRAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(f"Unexpected frame filename: {filename}")
    return int(match.group(1))


def prepare_single_image_project(
    project_root: Path,
    output_dir: Path,
    image_path: Path,
    canonical_frames_root: Path,
    suggestions_root: Path,
    class_template_path: Path | None = None,
) -> dict[str, Any]:
    """Create a standalone annotation project containing exactly one image."""
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    frame_index = parse_frame_index(image_path.name)
    video_id = image_path.name.rsplit("_frame_", 1)[0]
    manifest_path = output_dir / "project.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path, {})
        project_frames = [
            frame["filename"]
            for video in manifest.get("videos", [])
            for frame in video.get("frames", [])
        ]
        if project_frames != [image_path.name]:
            raise ValueError(
                "The single-image project already contains another image. "
                "Choose a different --project directory."
            )
        return manifest

    try:
        split, _ = discover_video(canonical_frames_root.resolve(), video_id)
        split_policy = "Preserve the established osce-yolo-auto video split."
    except ValueError:
        split = "train"
        split_policy = "Standalone image assigned to train because no canonical split was found."

    classes: list[dict[str, Any]] = []
    if class_template_path is not None and class_template_path.is_file():
        template = read_json(class_template_path, {})
        classes = json.loads(json.dumps(template.get("classes", [])))

    destination = output_dir / "images" / split / video_id / image_path.name
    materialization = link_image(image_path, destination)
    interval = 2.0
    manifest = {
        "schema_version": "1.0",
        "project_name": f"Single-image annotation — {image_path.name}",
        "description": "Standalone manual bounding-box annotation project.",
        "created_at": utc_now(),
        "split_policy": split_policy,
        "frames_root": relative_or_absolute(output_dir / "images", project_root),
        "suggestions_root": relative_or_absolute(suggestions_root, project_root),
        "image_materialization": [materialization],
        "classes": classes,
        "videos": [
            {
                "video_id": video_id,
                "split": split,
                "station": "",
                "source_relative_path": relative_or_absolute(image_path, project_root),
                "duration_seconds": round(frame_index * interval, 3),
                "frame_interval_seconds": interval,
                "frame_count": 1,
                "frames": [
                    {
                        "filename": image_path.name,
                        "index": frame_index,
                        "timestamp_seconds": round((frame_index - 1) * interval, 3),
                    }
                ],
            }
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        output_dir / "annotations.json",
        {"schema_version": "1.0", "updated_at": utc_now(), "frames": {}},
    )
    update_training_files(output_dir, manifest, {"frames": {}})
    return manifest


def prepare_annotation_project(
    project_root: Path,
    output_dir: Path,
    frames_root: Path,
    suggestions_root: Path,
    video_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create or reopen a three-video manual annotation project."""
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    frames_root = frames_root.resolve()
    suggestions_root = suggestions_root.resolve()
    manifest_path = output_dir / "project.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path, {})
        if video_ids and video_ids != [row["video_id"] for row in manifest["videos"]]:
            raise ValueError(
                "This annotation project already has a different video selection. "
                "Use another --project directory to start a new selection."
            )
        if not (output_dir / "annotations.json").is_file():
            atomic_write_json(
                output_dir / "annotations.json",
                {"schema_version": "1.0", "updated_at": utc_now(), "frames": {}},
            )
        return manifest

    selected = video_ids or choose_representative_videos(frames_root, suggestions_root)
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("Exactly three distinct video IDs are required for this pilot.")

    videos: list[dict[str, Any]] = []
    materialization_modes: set[str] = set()
    for video_id in selected:
        if not VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError(f"Invalid video ID: {video_id}")
        split, source_dir = discover_video(frames_root, video_id)
        metadata = read_json(source_dir / "metadata.json", {})
        interval = float(metadata.get("frame_interval_seconds", 2.0))
        image_paths = sorted(source_dir.glob("*.jpg"), key=lambda p: parse_frame_index(p.name))
        if not image_paths:
            raise ValueError(f"No JPEG frames in {source_dir}")
        frames: list[dict[str, Any]] = []
        for source in image_paths:
            index = parse_frame_index(source.name)
            destination = output_dir / "images" / split / video_id / source.name
            materialization_modes.add(link_image(source, destination))
            frames.append(
                {
                    "filename": source.name,
                    "index": index,
                    "timestamp_seconds": round((index - 1) * interval, 3),
                }
            )
        videos.append(
            {
                "video_id": video_id,
                "split": split,
                "station": metadata.get("station", ""),
                "source_relative_path": metadata.get("source_relative_path", ""),
                "duration_seconds": metadata.get(
                    "duration_seconds", frames[-1]["timestamp_seconds"]
                ),
                "frame_interval_seconds": interval,
                "frame_count": len(frames),
                "frames": frames,
            }
        )

    manifest = {
        "schema_version": "1.0",
        "project_name": "Catheterization OSCE — Manual Bounding Boxes",
        "description": "Three-video pilot with the established leakage-safe video split.",
        "created_at": utc_now(),
        "split_policy": "Preserve osce-yolo-auto video-level split (18 train / 6 dev / 6 test).",
        "frames_root": relative_or_absolute(frames_root, project_root),
        "suggestions_root": relative_or_absolute(suggestions_root, project_root),
        "image_materialization": sorted(materialization_modes),
        "classes": [],
        "videos": videos,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        output_dir / "annotations.json",
        {"schema_version": "1.0", "updated_at": utc_now(), "frames": {}},
    )
    update_training_files(output_dir, manifest, {"frames": {}})
    return manifest


def frame_key(video_id: str, filename: str) -> str:
    return f"{video_id}/{filename}"


def build_frame_index(
    manifest: dict[str, Any], frames_root: Path
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for video in manifest["videos"]:
        video_id = video["video_id"]
        split = video["split"]
        for frame in video["frames"]:
            key = frame_key(video_id, frame["filename"])
            source = frames_root / split / video_id / frame["filename"]
            if not source.is_file():
                raise FileNotFoundError(source)
            values[key] = {
                **frame,
                "key": key,
                "video_id": video_id,
                "split": split,
                "source": source,
            }
    return values


def parse_yolo_suggestions(
    path: Path, classes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not path.is_file():
        return boxes
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:5])
        except ValueError:
            continue
        if not 0 <= class_id < len(classes) or width <= 0 or height <= 0:
            continue
        x = max(0.0, center_x - width / 2)
        y = max(0.0, center_y - height / 2)
        width = min(width, 1.0 - x)
        height = min(height, 1.0 - y)
        if width <= 0 or height <= 0:
            continue
        boxes.append(
            {
                "id": f"suggestion-{line_number}",
                "class_id": class_id,
                "x": round(x, 6),
                "y": round(y, 6),
                "width": round(width, 6),
                "height": round(height, 6),
            }
        )
    return boxes


def validate_boxes(
    raw_boxes: Any, classes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(raw_boxes, list) or len(raw_boxes) > 100:
        raise ValueError("boxes must be a list with at most 100 entries")
    boxes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_boxes:
        if not isinstance(raw, dict):
            raise ValueError("Each box must be an object")
        box_id = str(raw.get("id", ""))
        if not BOX_ID_RE.fullmatch(box_id) or box_id in seen_ids:
            raise ValueError("Each box needs a unique safe ID")
        seen_ids.add(box_id)
        try:
            class_id = int(raw["class_id"])
            x = float(raw["x"])
            y = float(raw["y"])
            width = float(raw["width"])
            height = float(raw["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid box coordinates") from exc
        if not 0 <= class_id < len(classes):
            raise ValueError("Invalid class ID")
        if not all(map(lambda value: value == value and abs(value) != float("inf"), (x, y, width, height))):
            raise ValueError("Box coordinates must be finite")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Box coordinates must be positive and normalized")
        if x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("Box extends outside the image")
        if width < 0.001 or height < 0.001:
            raise ValueError("Box is too small")
        boxes.append(
            {
                "id": box_id,
                "class_id": class_id,
                "x": round(max(0.0, x), 6),
                "y": round(max(0.0, y), 6),
                "width": round(min(width, 1.0 - x), 6),
                "height": round(min(height, 1.0 - y), 6),
            }
        )
    return boxes


def yolo_line(box: dict[str, Any]) -> str:
    center_x = box["x"] + box["width"] / 2
    center_y = box["y"] + box["height"] / 2
    return (
        f"{box['class_id']} {center_x:.6f} {center_y:.6f} "
        f"{box['width']:.6f} {box['height']:.6f}"
    )


def update_training_files(
    output_dir: Path, manifest: dict[str, Any], store: dict[str, Any]
) -> None:
    """Mirror confirmed records to YOLO labels and confirmed-only split lists."""
    lookup = {
        frame_key(video["video_id"], frame["filename"]): (video, frame)
        for video in manifest["videos"]
        for frame in video["frames"]
    }
    confirmed_by_split: dict[str, list[Path]] = {split: [] for split in VALID_SPLITS}
    records = store.get("frames", {})
    for key, record in records.items():
        if key not in lookup:
            continue
        video, frame = lookup[key]
        split = video["split"]
        video_id = video["video_id"]
        label_path = output_dir / "labels" / split / video_id / (
            Path(frame["filename"]).stem + ".txt"
        )
        if record.get("status") in {"annotated", "empty"}:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            text = "\n".join(yolo_line(box) for box in record.get("boxes", []))
            if text:
                text += "\n"
            temporary = label_path.with_suffix(".txt.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(label_path)
            image_path = output_dir / "images" / split / video_id / frame["filename"]
            if image_path.exists():
                confirmed_by_split[split].append(image_path.resolve())
        elif label_path.exists():
            label_path.unlink()

    lists_dir = output_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    for split in VALID_SPLITS:
        paths = sorted(confirmed_by_split[split], key=str)
        (lists_dir / f"{split}.txt").write_text(
            "".join(f"{path.as_posix()}\n" for path in paths), encoding="utf-8"
        )
    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: lists/train.txt",
        "val: lists/dev.txt",
        "test: lists/test.txt",
    ]
    classes = manifest.get("classes", [])
    if classes:
        yaml_lines.append("names:")
        yaml_lines.extend(f"  {row['id']}: {row['name']}" for row in classes)
    else:
        yaml_lines.append("names: {}")
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def summarize(manifest: dict[str, Any], store: dict[str, Any]) -> dict[str, Any]:
    total = sum(len(video["frames"]) for video in manifest["videos"])
    counts = {status: 0 for status in VALID_STATUSES}
    classes = manifest.get("classes", [])
    box_counts = [0] * len(classes)
    for record in store.get("frames", {}).values():
        status = record.get("status", "unreviewed")
        if status in counts and status != "unreviewed":
            counts[status] += 1
        for box in record.get("boxes", []):
            class_id = box.get("class_id")
            if isinstance(class_id, int) and 0 <= class_id < len(box_counts):
                box_counts[class_id] += 1
    explicit = sum(counts[status] for status in ("annotated", "empty", "needs_review"))
    counts["unreviewed"] = total - explicit
    complete = counts["annotated"] + counts["empty"]
    return {
        "total_frames": total,
        "complete_frames": complete,
        "progress": round(complete / total, 6) if total else 0.0,
        "status_counts": counts,
        "box_counts": [
            {"class_id": index, "class_name": classes[index]["name"], "count": value}
            for index, value in enumerate(box_counts)
        ],
    }


def public_video_frames(video: dict[str, Any], records: dict[str, Any]) -> list[dict[str, Any]]:
    """Only frame metadata goes to the browser; box records load on demand."""
    frames: list[dict[str, Any]] = []
    for frame in video["frames"]:
        key = frame_key(video["video_id"], frame["filename"])
        record = records.get(key, {})
        frames.append(
            {
                **frame,
                "key": key,
                "status": record.get("status", "unreviewed"),
                "box_count": len(record.get("boxes", [])),
                "revision": int(record.get("revision", 0)),
            }
        )
    return frames


def public_project_payload(
    manifest: dict[str, Any], store: dict[str, Any], suggestions_root: Path
) -> dict[str, Any]:
    records = store.get("frames", {})
    videos: list[dict[str, Any]] = []
    for index, video in enumerate(manifest["videos"]):
        public_video = {key: value for key, value in video.items() if key != "frames"}
        public_video["complete_count"] = sum(
            records.get(frame_key(video["video_id"], frame["filename"]), {}).get("status")
            in {"annotated", "empty"}
            for frame in video["frames"]
        )
        # Load only the first video at startup. A 30-video project otherwise
        # sends thousands of frame records before the first image can render.
        public_video["frames"] = public_video_frames(video, records) if index == 0 else []
        public_video["frames_loaded"] = index == 0
        videos.append(public_video)
    return {
        **{key: value for key, value in manifest.items() if key != "videos"},
        "videos": videos,
        "summary": summarize(manifest, store),
    }


def export_archive(
    output_dir: Path, manifest: dict[str, Any], store: dict[str, Any]
) -> bytes:
    buffer = io.BytesIO()
    confirmed = {
        key: record
        for key, record in store.get("frames", {}).items()
        if record.get("status") in {"annotated", "empty"}
    }
    split_rows: dict[str, list[str]] = {split: [] for split in VALID_SPLITS}
    video_lookup = {video["video_id"]: video for video in manifest["videos"]}
    classes = manifest.get("classes", [])
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(
        [
            "split",
            "video_id",
            "filename",
            "timestamp_seconds",
            "status",
            "class_id",
            "class_name",
            "x_center",
            "y_center",
            "width",
            "height",
            "note",
        ]
    )
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, record in sorted(confirmed.items()):
            video_id, filename = key.split("/", 1)
            video = video_lookup[video_id]
            split = video["split"]
            frame = next(row for row in video["frames"] if row["filename"] == filename)
            label_name = f"labels/{split}/{video_id}/{Path(filename).stem}.txt"
            label_text = "\n".join(yolo_line(box) for box in record.get("boxes", []))
            archive.writestr(label_name, label_text + ("\n" if label_text else ""))
            split_rows[split].append(f"images/{split}/{video_id}/{filename}")
            if not record.get("boxes"):
                writer.writerow(
                    [
                        split,
                        video_id,
                        filename,
                        frame["timestamp_seconds"],
                        record["status"],
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        record.get("note", ""),
                    ]
                )
            for box in record.get("boxes", []):
                writer.writerow(
                    [
                        split,
                        video_id,
                        filename,
                        frame["timestamp_seconds"],
                        record["status"],
                        box["class_id"],
                        classes[box["class_id"]]["name"],
                        round(box["x"] + box["width"] / 2, 6),
                        round(box["y"] + box["height"] / 2, 6),
                        box["width"],
                        box["height"],
                        record.get("note", ""),
                    ]
                )
        for split, paths in split_rows.items():
            # Ultralytics resolves "./" entries relative to the list file.
            archive.writestr(f"{split}.txt", "".join(f"./{path}\n" for path in paths))
        yaml_lines: list[str] = [
            "path: .",
            "train: train.txt",
            "val: dev.txt",
            "test: test.txt",
        ]
        if classes:
            yaml_lines.append("names:")
            yaml_lines.extend(f"  {row['id']}: {row['name']}" for row in classes)
        else:
            yaml_lines.append("names: {}")
        archive.writestr("data.yaml", "\n".join(yaml_lines) + "\n")
        archive.writestr("annotations.csv", csv_buffer.getvalue())
        archive.writestr(
            "annotations.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "exported_at": utc_now(),
                    "frames": confirmed,
                    "summary": summarize(manifest, store),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        archive.writestr(
            "project.json", json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        archive.writestr(
            "README.md",
            "# Catheterization OSCE manual annotations\n\n"
            "Only manually confirmed frames are exported. The archive omits JPEG files "
            "to stay small; copy or hard-link the matching source frames into the listed "
            "`images/{split}/{video_id}` paths before training on another machine. The "
            "local annotation project is already training-ready through `data.yaml`.\n",
        )
    return buffer.getvalue()


def reset_project_classes(output_dir: Path) -> dict[str, Any]:
    """Clear a project's taxonomy only when no manual boxes would be orphaned."""
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "project.json"
    annotations_path = output_dir / "annotations.json"
    manifest = read_json(manifest_path, None)
    if not isinstance(manifest, dict):
        raise FileNotFoundError(f"Annotation project not found: {manifest_path}")
    store = read_json(annotations_path, {"frames": {}})
    box_count = sum(
        len(record.get("boxes", [])) for record in store.get("frames", {}).values()
    )
    if box_count:
        raise ValueError(
            f"Cannot reset classes because {box_count} manual boxes already use them."
        )
    manifest["classes"] = []
    manifest["taxonomy_updated_at"] = utc_now()
    atomic_write_json(manifest_path, manifest)
    update_training_files(output_dir, manifest, store)
    return manifest


class AnnotationServer(ThreadingHTTPServer):
    project_root: Path
    output_dir: Path
    manifest: dict[str, Any]
    frames_root: Path
    suggestions_root: Path
    frame_index: dict[str, dict[str, Any]]
    static_dir: Path
    annotation_lock: threading.Lock


class AnnotationHandler(BaseHTTPRequestHandler):
    server: AnnotationServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[annotator] {self.address_string()} - {format % args}")

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(
        self, body: bytes, content_type: str, filename: str | None = None
    ) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def serve_static(self, name: str) -> None:
        allowed = {"annotator.html", "annotator.css", "annotator.js"}
        if name not in allowed:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.server.static_dir / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), f"{content_type}; charset=utf-8")

    def query_frame_key(self) -> str | None:
        query = parse_qs(urlparse(self.path).query)
        video_id = query.get("video", [""])[0]
        filename = query.get("frame", [""])[0]
        key = frame_key(video_id, filename)
        if key not in self.server.frame_index:
            self.send_json({"error": "Unknown frame."}, HTTPStatus.NOT_FOUND)
            return None
        return key

    def serve_frame(self, head_only: bool = False) -> None:
        key = self.query_frame_key()
        if key is None:
            return
        path = self.server.frame_index[key]["source"]
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        if not head_only:
            try:
                with path.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

    def read_request_json(self) -> dict[str, Any] | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = -1
        if size < 1 or size > 1024 * 1024:
            self.send_json({"error": "Invalid request size."}, HTTPStatus.BAD_REQUEST)
            return None
        try:
            value = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(value, dict):
            self.send_json({"error": "JSON object required."}, HTTPStatus.BAD_REQUEST)
            return None
        return value

    def save_annotation(self) -> None:
        value = self.read_request_json()
        if value is None:
            return
        video_id = str(value.get("video_id", ""))
        filename = str(value.get("filename", ""))
        key = frame_key(video_id, filename)
        if key not in self.server.frame_index:
            self.send_json({"error": "Unknown frame."}, HTTPStatus.NOT_FOUND)
            return
        status = str(value.get("status", "unreviewed"))
        note = str(value.get("note", "")).strip()
        if status not in VALID_STATUSES:
            self.send_json({"error": "Invalid frame status."}, HTTPStatus.BAD_REQUEST)
            return
        if len(note) > 2000:
            self.send_json({"error": "Note is too long."}, HTTPStatus.BAD_REQUEST)
            return
        try:
            boxes = validate_boxes(
                value.get("boxes", []), self.server.manifest.get("classes", [])
            )
            expected_revision = int(value.get("expected_revision", 0))
        except (ValueError, TypeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if status == "annotated" and not boxes:
            self.send_json(
                {"error": "Use No objects for a confirmed frame without boxes."},
                HTTPStatus.BAD_REQUEST,
            )
            return
        if status == "empty" and boxes:
            self.send_json(
                {"error": "A No objects frame cannot contain boxes."},
                HTTPStatus.BAD_REQUEST,
            )
            return

        store_path = self.server.output_dir / "annotations.json"
        with self.server.annotation_lock:
            store = read_json(
                store_path,
                {"schema_version": "1.0", "updated_at": utc_now(), "frames": {}},
            )
            current = store.setdefault("frames", {}).get(key, {})
            current_revision = int(current.get("revision", 0))
            if expected_revision != current_revision:
                self.send_json(
                    {
                        "error": "This frame changed in another tab. Reload it before saving.",
                        "code": "revision_conflict",
                        "record": current,
                    },
                    HTTPStatus.CONFLICT,
                )
                return
            record = {
                "video_id": video_id,
                "filename": filename,
                "split": self.server.frame_index[key]["split"],
                "status": status,
                "boxes": boxes,
                "note": note,
                "revision": current_revision + 1,
                "updated_at": utc_now(),
            }
            store["frames"][key] = record
            store["updated_at"] = record["updated_at"]
            atomic_write_json(store_path, store)
            update_training_files(self.server.output_dir, self.server.manifest, store)
            summary = summarize(self.server.manifest, store)
        self.send_json({"ok": True, "record": record, "summary": summary})

    def add_class(self) -> None:
        value = self.read_request_json()
        if value is None:
            return
        label = str(value.get("label", ""))
        requested_color = str(value.get("color", "")).strip()
        with self.server.annotation_lock:
            classes = self.server.manifest.setdefault("classes", [])
            color = requested_color or CLASS_COLOR_PALETTE[
                len(classes) % len(CLASS_COLOR_PALETTE)
            ]
            try:
                new_class = build_class(label, color, classes)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            classes.append(new_class)
            self.server.manifest["taxonomy_updated_at"] = utc_now()
            atomic_write_json(
                self.server.output_dir / "project.json", self.server.manifest
            )
            store = read_json(
                self.server.output_dir / "annotations.json", {"frames": {}}
            )
            update_training_files(
                self.server.output_dir, self.server.manifest, store
            )
        self.send_json(
            {
                "ok": True,
                "class": new_class,
                "classes": classes,
                "suggestions_compatible": suggestions_are_compatible(
                    self.server.manifest
                ),
            },
            HTTPStatus.CREATED,
        )

    def do_HEAD(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/media/annotation-frame":
            self.serve_frame(head_only=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/annotation-project":
            with self.server.annotation_lock:
                store = read_json(self.server.output_dir / "annotations.json", {"frames": {}})
                payload = public_project_payload(
                    self.server.manifest, store, self.server.suggestions_root
                )
            self.send_json(payload)
        elif path == "/api/annotation-video":
            video_id = parse_qs(urlparse(self.path).query).get("video", [""])[0]
            video = next((row for row in self.server.manifest["videos"] if row["video_id"] == video_id), None)
            if video is None:
                self.send_json({"error": "Unknown video."}, HTTPStatus.NOT_FOUND)
                return
            with self.server.annotation_lock:
                store = read_json(self.server.output_dir / "annotations.json", {"frames": {}})
                frames = public_video_frames(video, store.get("frames", {}))
            self.send_json({
                "video_id": video_id,
                "frames": frames,
                "complete_count": sum(frame["status"] in {"annotated", "empty"} for frame in frames),
            })
        elif path == "/api/annotation-frame":
            key = self.query_frame_key()
            if key is None:
                return
            info = self.server.frame_index[key]
            suggestion_path = (
                self.server.suggestions_root
                / info["split"]
                / info["video_id"]
                / f"{Path(info['filename']).stem}.txt"
            )
            with self.server.annotation_lock:
                store = read_json(self.server.output_dir / "annotations.json", {"frames": {}})
                record = store.get("frames", {}).get(key)
            compatible = suggestions_are_compatible(self.server.manifest)
            self.send_json(
                {
                    "key": key,
                    "record": record,
                    "suggestions": (
                        parse_yolo_suggestions(
                            suggestion_path, self.server.manifest.get("classes", [])
                        )
                        if compatible
                        else []
                    ),
                    "image_url": (
                        "/media/annotation-frame?video="
                        f"{info['video_id']}&frame={info['filename']}"
                    ),
                }
            )
        elif path == "/api/annotation-export":
            with self.server.annotation_lock:
                store = read_json(self.server.output_dir / "annotations.json", {"frames": {}})
                body = export_archive(self.server.output_dir, self.server.manifest, store)
            self.send_bytes(
                body,
                "application/zip",
                "catheterization_manual_annotations.zip",
            )
        elif path == "/media/annotation-frame":
            self.serve_frame()
        elif path in {"/", "/annotator", "/annotator.html"}:
            self.serve_static("annotator.html")
        elif path in {"/annotator.css", "/annotator.js"}:
            self.serve_static(path[1:])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/annotation-save":
            self.save_annotation()
        elif path == "/api/annotation-class":
            self.add_class()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def create_server(
    project_root: Path,
    output_dir: Path,
    frames_root: Path,
    suggestions_root: Path,
    video_ids: list[str] | None = None,
    host: str = "127.0.0.1",
    port: int = 8770,
) -> AnnotationServer:
    project_root = project_root.resolve()
    manifest = prepare_annotation_project(
        project_root, output_dir, frames_root, suggestions_root, video_ids
    )
    manifest_frames_root = resolve_project_path(manifest["frames_root"], project_root)
    manifest_suggestions_root = resolve_project_path(
        manifest["suggestions_root"], project_root
    )
    server = AnnotationServer((host, port), AnnotationHandler)
    server.project_root = project_root
    server.output_dir = output_dir.resolve()
    server.manifest = manifest
    server.frames_root = manifest_frames_root
    server.suggestions_root = manifest_suggestions_root
    server.frame_index = build_frame_index(manifest, manifest_frames_root)
    server.static_dir = Path(__file__).with_name("web")
    server.annotation_lock = threading.Lock()
    return server


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--frames-root", type=Path, default=project_root / DEFAULT_FRAMES_RELATIVE
    )
    parser.add_argument(
        "--suggestions-root",
        type=Path,
        default=project_root / DEFAULT_SUGGESTIONS_RELATIVE,
    )
    parser.add_argument(
        "--videos",
        nargs=3,
        metavar=("VIDEO_1", "VIDEO_2", "VIDEO_3"),
        help="Exactly three video IDs; their established splits are preserved.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Create or reopen a project containing only this JPEG frame.",
    )
    parser.add_argument(
        "--class-template",
        type=Path,
        default=project_root / DEFAULT_OUTPUT_RELATIVE / "project.json",
        help="Project manifest whose class list is copied for a new single-image project.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--open", action="store_true", help="Open the browser")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Create the project without serving"
    )
    parser.add_argument(
        "--reset-classes",
        action="store_true",
        help="Clear the current taxonomy if the project contains no manual boxes",
    )
    args = parser.parse_args(argv)
    if args.image and args.videos:
        parser.error("--image and --videos cannot be used together")
    output_dir = args.project or (
        project_root / "data/catheterization/manual-annotation-single-image"
        if args.image
        else project_root / DEFAULT_OUTPUT_RELATIVE
    )
    if args.image:
        prepare_single_image_project(
            project_root,
            output_dir,
            args.image,
            args.frames_root,
            args.suggestions_root,
            args.class_template,
        )
    if args.reset_classes and (output_dir / "project.json").is_file():
        reset_project_classes(output_dir)
        print(f"Classes reset: {output_dir.resolve()}")
    server = create_server(
        project_root,
        output_dir,
        args.frames_root,
        args.suggestions_root,
        args.videos,
        args.host,
        args.port,
    )
    if args.prepare_only:
        print(f"Annotation project prepared: {server.output_dir}")
        server.server_close()
        return
    url = f"http://{args.host}:{server.server_port}/annotator"
    print(f"SkilLens annotator: {url}")
    print(f"Project: {server.output_dir}")
    print("Split: preserved from osce-yolo-auto (video-level, leakage-safe)")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotator.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
