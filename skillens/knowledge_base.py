"""File-based rubric knowledge base with no Python exam registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base" / "rubrics"


def _resolve_reference(value: str, rubric_path: Path) -> Path:
    if value.startswith("project://"):
        return (PROJECT_ROOT / value.removeprefix("project://")).resolve()
    return (rubric_path.parent / value).resolve()


def normalize_rubric(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    framework = dict(raw.get("framework") or {})
    exam_id = str(framework.get("exam_id") or path.stem).strip()
    if not exam_id or not exam_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid exam_id {exam_id!r} in {path}")
    steps = raw.get("steps") or raw.get("criteria")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Rubric must contain a non-empty steps list: {path}")
    normalized_steps = []
    seen: set[int] = set()
    for position, value in enumerate(steps, start=1):
        step = dict(value)
        step_id = int(step.get("id", position))
        if step_id in seen:
            raise ValueError(f"Duplicate step ID {step_id} in {path}")
        seen.add(step_id)
        description = str(step.get("description") or "").strip()
        if not description:
            raise ValueError(f"Step {step_id} has no description in {path}")
        allowed = sorted({int(score) for score in step.get("allowed_scores", [0, 1, 2])})
        max_score = int(step.get("max_score", max(allowed)))
        if (
            not allowed
            or not set(allowed).issubset({0, 1, 2})
            or min(allowed) < 0
            or max(allowed) > max_score
            or max_score > 2
        ):
            raise ValueError(f"Invalid allowed_scores for step {step_id}: {allowed}")
        step.update(
            {
                "id": step_id,
                "key": str(step.get("key") or f"criterion_{step_id}"),
                "description": description,
                "allowed_scores": allowed,
                "max_score": max_score,
                "modalities": list(step.get("modalities") or ["video"]),
                "required_objects": list(step.get("required_objects") or []),
                "observable_evidence": list(step.get("observable_evidence") or []),
            }
        )
        normalized_steps.append(step)
    normalized_steps.sort(key=lambda item: item["id"])
    maximum = sum(step["max_score"] for step in normalized_steps)
    declared = int(raw.get("maximum_score", raw.get("maximum_raw_score", maximum)))
    if declared != maximum:
        raise ValueError(f"Maximum score mismatch in {path}: declared={declared}, steps={maximum}")
    result = dict(raw)
    backend = str(framework.get("default_backend", "full_api"))
    if backend not in {"yolo", "sam31", "full_api"}:
        raise ValueError(f"Invalid default_backend {backend!r} in {path}")
    audio_mode = str(framework.get("audio", "auto"))
    if audio_mode not in {"auto", "always", "never"}:
        raise ValueError(f"Invalid audio mode {audio_mode!r} in {path}")
    result.update(
        {
            "framework": {**framework, "exam_id": exam_id},
            "rubric_id": str(raw.get("rubric_id") or f"{exam_id}_v1"),
            "title": str(raw.get("title") or exam_id.replace("_", " ").title()),
            "maximum_score": maximum,
            "score_scale": raw.get("score_scale")
            or {"0": "incorrect", "1": "partial", "2": "complete"},
            "steps": normalized_steps,
        }
    )
    result.pop("criteria", None)
    return result


@dataclass(frozen=True)
class ExamSpec:
    exam_id: str
    rubric_path: Path
    rubric: dict[str, Any]

    @property
    def framework(self) -> dict[str, Any]:
        return self.rubric["framework"]

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.framework.get("aliases", []))

    @property
    def default_backend(self) -> str:
        return str(self.framework.get("default_backend", "full_api"))

    @property
    def requires_audio(self) -> bool:
        return any("audio" in step["modalities"] for step in self.rubric["steps"])

    def config(self, key: str, default: Any = None) -> Any:
        return self.framework.get(key, default)

    def asset_value(self, key: str, default: Any = None) -> Any:
        return self.framework.get("assets", {}).get(key, default)

    def asset_path(self, key: str) -> Path | None:
        value = self.asset_value(key)
        return _resolve_reference(str(value), self.rubric_path) if value else None

    def gold_path(self) -> Path | None:
        value = self.framework.get("gold", {}).get("path")
        return _resolve_reference(str(value), self.rubric_path) if value else None


class RubricKnowledgeBase:
    def __init__(self, directories: Iterable[Path] | None = None):
        self.directories = tuple(
            path.expanduser().resolve()
            for path in (directories or (DEFAULT_KNOWLEDGE_BASE,))
        )
        self._exams = self._scan()

    def _scan(self) -> dict[str, ExamSpec]:
        exams: dict[str, ExamSpec] = {}
        aliases: dict[str, str] = {}
        for directory in self.directories:
            if not directory.is_dir():
                raise FileNotFoundError(f"Knowledge-base directory not found: {directory}")
            for path in sorted(directory.glob("*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                # A knowledge-base directory may also hold auxiliary JSON files.
                # Only rubric-shaped documents participate in discovery.
                if not isinstance(raw, dict) or not raw.get("steps") or not any(
                    key in raw for key in ("framework", "rubric_id", "title")
                ):
                    continue
                rubric = normalize_rubric(raw, path)
                exam_id = rubric["framework"]["exam_id"]
                if exam_id in exams:
                    raise ValueError(f"Duplicate exam_id {exam_id!r}: {path}")
                spec = ExamSpec(exam_id, path.resolve(), rubric)
                exams[exam_id] = spec
                for alias in (exam_id, path.stem, rubric["rubric_id"], *spec.aliases):
                    key = str(alias).casefold()
                    if key in aliases and aliases[key] != exam_id:
                        raise ValueError(f"Duplicate exam alias {alias!r}")
                    aliases[key] = exam_id
        if not exams:
            raise ValueError("The rubric knowledge base is empty")
        self._aliases = aliases
        return exams

    def list(self) -> list[ExamSpec]:
        return [self._exams[key] for key in sorted(self._exams)]

    def get(self, identifier: str) -> ExamSpec:
        exam_id = self._aliases.get(identifier.casefold())
        if exam_id is None:
            choices = ", ".join(sorted(self._exams))
            raise KeyError(f"Unknown exam {identifier!r}. Available exams: {choices}")
        return self._exams[exam_id]

    def validate_assets(self, spec: ExamSpec) -> list[str]:
        warnings = []
        for key in ("yolo_model", "sam_manifest", "sam_manifest_video"):
            path = spec.asset_path(key)
            if path is not None and not path.is_file():
                warnings.append(f"{key} not found: {path}")
        gold = spec.gold_path()
        if gold is not None and not gold.is_file():
            warnings.append(f"gold workbook not found: {gold}")
        return warnings
