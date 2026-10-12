"""Local review dashboard for timestamped SkilLens assessment results."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


BYTE_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
EXPERT_SAMPLE_ID_RE = re.compile(r"ER-\d{3}$")
EXPERT_REVIEW_RELATIVE = Path(
    "data/catheterization/osce-yolo-full6-silver-v4/expert-review"
)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(project_root: Path, exam: str) -> list[Path]:
    output_root = project_root / "output" / exam
    return sorted(
        (path.parent for path in output_root.glob("*/*/assessment.json")),
        key=lambda path: path.joinpath("assessment.json").stat().st_mtime,
        reverse=True,
    )


def resolve_run(project_root: Path, exam: str, requested: Path | None) -> Path:
    if requested is not None:
        run = requested.expanduser().resolve()
        if run.is_file() and run.name == "assessment.json":
            run = run.parent
        if not (run / "assessment.json").is_file():
            raise FileNotFoundError(f"No assessment.json in {run}")
        return run
    runs = discover_runs(project_root, exam)
    if not runs:
        raise FileNotFoundError(f"No completed runs found under output/{exam}")
    return runs[0]


def resolve_video(project_root: Path, exam: str, run: Path, assessment: dict[str, Any]) -> Path:
    raw = str(assessment.get("metadata", {}).get("source_video", ""))
    if raw:
        # Older artifacts serialized Windows separators twice.
        candidate = Path(raw.replace("\\\\", "\\"))
        if candidate.is_file():
            return candidate.resolve()

    candidate_name = run.parent.name
    exact = project_root / "data" / exam / f"{candidate_name}.mp4"
    if exact.is_file():
        return exact.resolve()
    videos = sorted((project_root / "data" / exam).glob("*.mp4"))
    if len(videos) == 1:
        return videos[0].resolve()
    raise FileNotFoundError(f"Unable to resolve source video for {run}")


def evidence_interval(evidence: dict[str, Any], event: dict[str, Any] | None) -> tuple[float, float]:
    seconds = [float(value) for value in evidence.get("evidence_seconds", [])]
    if seconds:
        return min(seconds), max(seconds)
    if event:
        return float(event["start_seconds"]), float(event["end_seconds"])
    return 0.0, 0.0


def build_dashboard(project_root: Path, exam: str = "catheterization", run_path: Path | None = None) -> tuple[dict[str, Any], Path]:
    run = resolve_run(project_root, exam, run_path)
    assessment = read_json(run / "assessment.json", {})
    observations = read_json(run / "observations.json", [])
    audio = read_json(run / "audio_evidence.json", {})
    rubric = read_json(run / "rubric.normalized.json", {})
    evaluation = read_json(run / "evaluation.json", {})
    assessment_images = read_json(run / "assessment_images.json", {})
    perception = read_json(run / "perception.json", {})
    video = resolve_video(project_root, exam, run, assessment)

    event_lookup = {event["event_id"]: event for event in observations}
    image_targets = {
        int(target["step_id"]): target for target in assessment_images.get("targets", [])
    }

    criteria = []
    for step in assessment.get("steps", []):
        evidence_rows = []
        for evidence in step.get("evidence", []):
            event = event_lookup.get(evidence.get("event_id"))
            start, end = evidence_interval(evidence, event)
            evidence_rows.append(
                {
                    "id": evidence.get("observation_id"),
                    "event_id": evidence.get("event_id"),
                    "status": evidence.get("status", "unclear"),
                    "confidence": evidence.get("confidence", 0.0),
                    "description": evidence.get("description", ""),
                    "start_seconds": start,
                    "end_seconds": end,
                    "event_start_seconds": event.get("start_seconds") if event else start,
                    "event_end_seconds": event.get("end_seconds") if event else end,
                    "evidence_seconds": evidence.get("evidence_seconds", []),
                    "source": "visual",
                }
            )
        for evidence in step.get("audio_evidence", []):
            evidence_rows.append(
                {
                    "id": evidence.get("audio_id"),
                    "event_id": None,
                    "status": evidence.get("status", "supporting"),
                    "confidence": evidence.get("confidence", 0.0),
                    "description": evidence.get("text", evidence.get("description", "")),
                    "start_seconds": float(evidence.get("start_seconds", 0.0)),
                    "end_seconds": float(evidence.get("end_seconds", 0.0)),
                    "event_start_seconds": float(evidence.get("start_seconds", 0.0)),
                    "event_end_seconds": float(evidence.get("end_seconds", 0.0)),
                    "evidence_seconds": [],
                    "source": "audio",
                }
            )
        evidence_rows.sort(key=lambda row: row["start_seconds"])
        criteria.append(
            {
                "step_id": int(step["step_id"]),
                "key": step.get("key"),
                "description": step.get("description", ""),
                "score": step.get("score"),
                "max_score": step.get("max_score", 2),
                "verdict": step.get("verdict", "unverifiable"),
                "confidence": step.get("confidence", 0.0),
                "score_basis": step.get("score_basis"),
                "reason": step.get("reason", ""),
                "requires_review": step.get("requires_review", False),
                "review_reasons": step.get("review_reasons", []),
                "evidence": evidence_rows,
                "review_images": image_targets.get(int(step["step_id"]), {}).get("timestamps", []),
            }
        )

    timeline_events = []
    for event in observations:
        event_observations = []
        for observation in event.get("observations", []):
            event_observations.append(
                {
                    "id": observation.get("observation_id"),
                    "step_id": observation.get("step_id"),
                    "status": observation.get("status", "unclear"),
                    "confidence": observation.get("confidence", 0.0),
                    "description": observation.get("description", ""),
                    "evidence_seconds": observation.get("evidence_seconds", []),
                }
            )
        decisive = [row for row in event_observations if row["status"] != "unclear"]
        decisive.sort(key=lambda row: float(row["confidence"]), reverse=True)
        timeline_events.append(
            {
                "event_id": event["event_id"],
                "start_seconds": float(event["start_seconds"]),
                "end_seconds": float(event["end_seconds"]),
                "sampled_timestamps": event.get("sampled_timestamps", []),
                "objects": event.get("objects", []),
                "observations": event_observations,
                "step_ids": sorted(
                    {int(row["step_id"]) for row in decisive if row.get("step_id") is not None}
                ),
                "headline": decisive[0]["description"] if decisive else "No decisive observation",
                "status": decisive[0]["status"] if decisive else "unclear",
            }
        )

    duration_candidates = [
        assessment_images.get("duration_seconds"),
        perception.get("metadata", {}).get("source_duration"),
        max((event["end_seconds"] for event in timeline_events), default=0.0),
    ]
    duration = max(float(value) for value in duration_candidates if value is not None)
    runs = discover_runs(project_root, exam)
    payload = {
        "schema_version": "1.0",
        "exam": exam,
        "title": rubric.get("title", "SkilLens assessment review"),
        "candidate": evaluation.get("candidate_id") or run.parent.name,
        "backend": run.name,
        "run_directory": str(run),
        "duration_seconds": duration,
        "video_url": "/media/video",
        "summary": assessment.get("summary", {}),
        "overall_feedback": assessment.get("overall_feedback", ""),
        "safety_concerns": assessment.get("safety_concerns", []),
        "sequence_violations": assessment.get("sequence_violations", []),
        "criteria": criteria,
        "timeline_events": timeline_events,
        "audio_chunks": audio.get("chunks", []),
        "available_runs": [
            {"candidate": path.parent.name, "backend": path.name, "path": str(path)} for path in runs
        ],
    }
    return payload, video


def build_expert_review(
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path]:
    """Load the balanced label-review manifest and resolve its image files."""
    review_root = (project_root / EXPERT_REVIEW_RELATIVE).resolve()
    manifest_path = review_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Expert-review manifest not found: {manifest_path}. "
            "Run tools/build_catheterization_expert_review_sample.py first."
        )
    manifest = read_json(manifest_path, {})
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"No samples in {manifest_path}")

    project_root = project_root.resolve()
    image_paths: dict[str, Path] = {}
    public_samples: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        if not EXPERT_SAMPLE_ID_RE.fullmatch(sample_id) or sample_id in image_paths:
            raise ValueError(f"Invalid or duplicate expert sample ID: {sample_id!r}")
        source = (project_root / str(sample.get("image_source", ""))).resolve()
        if project_root != source and project_root not in source.parents:
            raise ValueError(f"Expert image leaves project root: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        image_paths[sample_id] = source
        public = {key: value for key, value in sample.items() if key != "image_source"}
        public["image_url"] = f"/media/expert-review?sample={sample_id}"
        public_samples.append(public)

    reviews_path = review_root / "reviews.json"
    review_store = read_json(reviews_path, {"schema_version": "1.0", "reviews": {}})
    reviews = review_store.get("reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError(f"Invalid reviews object in {reviews_path}")
    payload = {
        key: value for key, value in manifest.items() if key != "samples"
    }
    payload["samples"] = public_samples
    payload["reviews"] = reviews
    payload["review_file"] = str(reviews_path)
    return payload, image_paths, reviews_path


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    if not header:
        return None
    match = BYTE_RANGE_RE.fullmatch(header.strip())
    if not match:
        raise ValueError("Malformed Range header")
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise ValueError("Empty Range header")
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("Invalid suffix length")
        start = max(0, size - length)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start >= size or start < 0 or end < start:
        raise ValueError("Unsatisfiable Range header")
    return start, min(end, size - 1)


class DashboardServer(ThreadingHTTPServer):
    dashboard: dict[str, Any]
    video_path: Path
    static_dir: Path
    expert_review: dict[str, Any]
    expert_image_paths: dict[str, Path]
    expert_reviews_path: Path
    expert_review_lock: threading.Lock


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static(self, name: str) -> None:
        allowed = {
            "index.html",
            "styles.css",
            "app.js",
            "expert_review.html",
            "expert_review.css",
            "expert_review.js",
        }
        if name not in allowed:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        path = self.server.static_dir / name
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def serve_expert_image(self, head_only: bool = False) -> None:
        query = parse_qs(urlparse(self.path).query)
        sample_id = query.get("sample", [""])[0]
        path = self.server.expert_image_paths.get(sample_id)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        if not head_only:
            try:
                self.wfile.write(path.read_bytes())
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

    def read_request_json(self) -> dict[str, Any] | None:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = -1
        if size < 1 or size > 64 * 1024:
            self.send_json(
                {"error": "Invalid request size."}, HTTPStatus.BAD_REQUEST
            )
            return None
        try:
            value = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
            return None
        if not isinstance(value, dict):
            self.send_json(
                {"error": "The request must be a JSON object."},
                HTTPStatus.BAD_REQUEST,
            )
            return None
        return value

    def save_expert_decision(self) -> None:
        value = self.read_request_json()
        if value is None:
            return
        sample_id = str(value.get("sample_id", ""))
        status = str(value.get("status", ""))
        note = str(value.get("note", "")).strip()
        reviewer = str(value.get("reviewer", "")).strip()
        if sample_id not in self.server.expert_image_paths:
            self.send_json({"error": "Unknown sample."}, HTTPStatus.NOT_FOUND)
            return
        if status not in {"pass", "needs_review", "unreviewed"}:
            self.send_json(
                {"error": "Status must be pass, needs_review, or unreviewed."},
                HTTPStatus.BAD_REQUEST,
            )
            return
        if len(note) > 2000 or len(reviewer) > 100:
            self.send_json(
                {"error": "The note or reviewer name is too long."},
                HTTPStatus.BAD_REQUEST,
            )
            return

        updated_at = datetime.now(UTC).isoformat()
        record = {
            "sample_id": sample_id,
            "status": status,
            "note": note,
            "reviewer": reviewer,
            "updated_at": updated_at,
        }
        with self.server.expert_review_lock:
            store = read_json(
                self.server.expert_reviews_path,
                {"schema_version": "1.0", "reviews": {}},
            )
            reviews = store.setdefault("reviews", {})
            if status == "unreviewed" and not note and not reviewer:
                reviews.pop(sample_id, None)
                record = None
            else:
                reviews[sample_id] = record
            store["updated_at"] = updated_at
            temporary = self.server.expert_reviews_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.server.expert_reviews_path)
            self.server.expert_review["reviews"] = reviews
        self.send_json({"ok": True, "review": record})

    def serve_video(self, head_only: bool = False) -> None:
        path = self.server.video_path
        size = path.stat().st_size
        try:
            selected = parse_byte_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = selected or (0, size - 1)
        content_length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if selected:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        if head_only:
            return
        remaining = content_length
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    # Browsers routinely cancel an in-flight range request
                    # after the user seeks to another timestamp.
                    break
                remaining -= len(chunk)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path == "/media/video":
            self.serve_video(head_only=True)
        elif path == "/media/expert-review":
            self.serve_expert_image(head_only=True)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/dashboard":
            self.send_json(self.server.dashboard)
        elif path == "/api/expert-review":
            with self.server.expert_review_lock:
                payload = dict(self.server.expert_review)
                store = read_json(
                    self.server.expert_reviews_path,
                    {"schema_version": "1.0", "reviews": {}},
                )
                payload["reviews"] = store.get("reviews", {})
            self.send_json(payload)
        elif path == "/media/video":
            self.serve_video()
        elif path == "/media/expert-review":
            self.serve_expert_image()
        elif path in {"/", "/index.html"}:
            self.serve_static("index.html")
        elif path in {"/expert-review", "/expert_review.html"}:
            self.serve_static("expert_review.html")
        elif path in {
            "/styles.css",
            "/app.js",
            "/expert_review.css",
            "/expert_review.js",
        }:
            self.serve_static(path[1:])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/expert-review/decision":
            self.save_expert_decision()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def create_server(
    project_root: Path,
    exam: str = "catheterization",
    run_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> DashboardServer:
    dashboard, video = build_dashboard(project_root, exam, run_path)
    expert_review, expert_images, expert_reviews_path = build_expert_review(
        project_root
    )
    server = DashboardServer((host, port), DashboardHandler)
    server.dashboard = dashboard
    server.video_path = video
    server.static_dir = Path(__file__).with_name("web")
    server.expert_review = expert_review
    server.expert_image_paths = expert_images
    server.expert_reviews_path = expert_reviews_path
    server.expert_review_lock = threading.Lock()
    return server


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", default="catheterization")
    parser.add_argument("--run", type=Path, help="Run directory or assessment.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the browser automatically")
    parser.add_argument(
        "--expert-review",
        action="store_true",
        help="Open the expert bounding-box review page",
    )
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    server = create_server(project_root, args.exam, args.run, args.host, args.port)
    route = "/expert-review" if args.expert_review else ""
    url = f"http://{args.host}:{server.server_port}{route}"
    print(f"SkilLens dashboard: {url}")
    print(f"Run: {server.dashboard['run_directory']}")
    print(f"Video: {server.video_path}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
