from __future__ import annotations

import io
import json
import threading
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from skillens.annotator import (
    create_server,
    prepare_annotation_project,
    prepare_single_image_project,
    reset_project_classes,
    validate_boxes,
)


VIDEO_IDS = (
    "video_0000000000000001",
    "video_0000000000000002",
    "video_0000000000000003",
)


def build_fixture(root: Path) -> tuple[Path, Path, Path]:
    frames = root / "frames"
    suggestions = root / "suggestions"
    output = root / "manual"
    for split, video_id in zip(("train", "dev", "test"), VIDEO_IDS, strict=True):
        video_dir = frames / split / video_id
        label_dir = suggestions / split / video_id
        video_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (video_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "split": split,
                    "station": f"Station {split}",
                    "duration_seconds": 4,
                    "frame_interval_seconds": 2,
                }
            ),
            encoding="utf-8",
        )
        for index in (1, 2):
            name = f"{video_id}_frame_{index:06d}"
            (video_dir / f"{name}.jpg").write_bytes(b"fake-jpeg")
            (label_dir / f"{name}.txt").write_text(
                "0 0.500000 0.500000 0.200000 0.400000\n", encoding="utf-8"
            )
    return frames, suggestions, output


def test_prepare_project_preserves_video_splits(tmp_path: Path) -> None:
    frames, suggestions, output = build_fixture(tmp_path)
    manifest = prepare_annotation_project(
        tmp_path, output, frames, suggestions, list(VIDEO_IDS)
    )
    assert [video["split"] for video in manifest["videos"]] == ["train", "dev", "test"]
    assert [video["frame_count"] for video in manifest["videos"]] == [2, 2, 2]
    assert manifest["videos"][0]["frames"][1]["timestamp_seconds"] == 2
    assert (output / "images/train" / VIDEO_IDS[0]).is_dir()
    assert (output / "data.yaml").is_file()
    assert manifest["classes"] == []
    assert "names: {}" in (output / "data.yaml").read_text(encoding="utf-8")


def test_validate_boxes_rejects_boxes_outside_image() -> None:
    with pytest.raises(ValueError, match="outside"):
        validate_boxes(
            [
                {
                    "id": "box-1",
                    "class_id": 0,
                    "x": 0.9,
                    "y": 0.2,
                    "width": 0.2,
                    "height": 0.2,
                }
            ],
            [{"id": 0, "name": "tool", "label": "Tool", "color": "#007C91"}],
        )


def test_prepare_single_image_project_inherits_classes(tmp_path: Path) -> None:
    frames, suggestions, template_output = build_fixture(tmp_path)
    template = prepare_annotation_project(
        tmp_path, template_output, frames, suggestions, list(VIDEO_IDS)
    )
    template["classes"] = [
        {"id": 0, "name": "forceps", "label": "Forceps", "color": "#007C91"}
    ]
    template_path = template_output / "project.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")
    image = frames / "train" / VIDEO_IDS[0] / f"{VIDEO_IDS[0]}_frame_000002.jpg"
    output = tmp_path / "single"

    manifest = prepare_single_image_project(
        tmp_path, output, image, frames, suggestions, template_path
    )

    assert manifest["videos"][0]["split"] == "train"
    assert manifest["videos"][0]["frame_count"] == 1
    assert manifest["videos"][0]["frames"][0]["index"] == 2
    assert manifest["classes"][0]["name"] == "forceps"
    linked = output / "images/train" / VIDEO_IDS[0] / image.name
    assert linked.samefile(image)


def test_annotation_http_save_and_export(tmp_path: Path) -> None:
    frames, suggestions, output = build_fixture(tmp_path)
    server = create_server(
        tmp_path,
        output,
        frames,
        suggestions,
        list(VIDEO_IDS),
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    filename = f"{VIDEO_IDS[0]}_frame_000001.jpg"
    try:
        with urlopen(f"{base}/api/annotation-project") as response:
            project = json.loads(response.read())
        assert project["summary"]["total_frames"] == 6
        assert "annotations" not in project
        assert "has_suggestion" not in project["videos"][0]["frames"][0]
        assert len(project["videos"][0]["frames"]) == 2
        assert project["videos"][1]["frames"] == []
        assert project["videos"][1]["frames_loaded"] is False

        video_query = urlencode({"video": VIDEO_IDS[1]})
        with urlopen(f"{base}/api/annotation-video?{video_query}") as response:
            video_frames = json.loads(response.read())
        assert len(video_frames["frames"]) == 2
        assert video_frames["complete_count"] == 0

        query = urlencode({"video": VIDEO_IDS[0], "frame": filename})
        with urlopen(f"{base}/api/annotation-frame?{query}") as response:
            frame = json.loads(response.read())
        assert frame["suggestions"] == []
        with urlopen(f"{base}{frame['image_url']}") as response:
            assert response.read() == b"fake-jpeg"

        class_body = json.dumps(
            {"label": "Urinary catheter", "color": "#007c91"}
        ).encode()
        class_request = Request(
            f"{base}/api/annotation-class",
            data=class_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(class_request) as response:
            class_result = json.loads(response.read())
        assert response.status == 201
        assert class_result["class"] == {
            "id": 0,
            "name": "urinary_catheter",
            "label": "Urinary catheter",
            "color": "#007C91",
        }

        body = json.dumps(
            {
                "video_id": VIDEO_IDS[0],
                "filename": filename,
                "status": "annotated",
                "note": "Checked manually",
                "expected_revision": 0,
                "boxes": [
                    {
                        "id": "box-1",
                        "class_id": 0,
                        "x": 0.4,
                        "y": 0.3,
                        "width": 0.2,
                        "height": 0.4,
                    }
                ],
            }
        ).encode()
        request = Request(
            f"{base}/api/annotation-save",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            saved = json.loads(response.read())
        assert saved["record"]["revision"] == 1
        assert saved["summary"]["complete_frames"] == 1
        label = output / "labels/train" / VIDEO_IDS[0] / filename.replace(".jpg", ".txt")
        assert label.read_text(encoding="utf-8") == (
            "0 0.500000 0.500000 0.200000 0.400000\n"
        )
        assert filename in (output / "lists/train.txt").read_text(encoding="utf-8")

        with urlopen(f"{base}/api/annotation-export") as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = set(archive.namelist())
            assert f"labels/train/{VIDEO_IDS[0]}/{filename[:-4]}.txt" in names
            assert "annotations.csv" in names
            assert "data.yaml" in names
            assert archive.read("train.txt").decode().startswith("./images/train/")
            assert "train: train.txt" in archive.read("data.yaml").decode()
            assert "0: urinary_catheter" in archive.read("data.yaml").decode()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reset_classes_refuses_to_orphan_boxes(tmp_path: Path) -> None:
    frames, suggestions, output = build_fixture(tmp_path)
    manifest = prepare_annotation_project(
        tmp_path, output, frames, suggestions, list(VIDEO_IDS)
    )
    manifest["classes"] = [
        {"id": 0, "name": "tool", "label": "Tool", "color": "#007C91"}
    ]
    (output / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "annotations.json").write_text(
        json.dumps(
            {
                "frames": {
                    "key": {
                        "boxes": [
                            {
                                "id": "box-1",
                                "class_id": 0,
                                "x": 0.1,
                                "y": 0.1,
                                "width": 0.2,
                                "height": 0.2,
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manual boxes"):
        reset_project_classes(output)
