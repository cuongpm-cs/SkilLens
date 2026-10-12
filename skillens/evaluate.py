"""Knowledge-base-driven evaluation for all registered exams."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import legacy_aseptic_gold, legacy_catheter_gold
from .knowledge_base import DEFAULT_KNOWLEDGE_BASE, ExamSpec, RubricKnowledgeBase


@dataclass(frozen=True)
class GoldScores:
    candidate_id: str
    scores: dict[int, int]


def _assessment_predictions(assessment: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for item in assessment.get("steps") or assessment.get("criteria") or []:
        raw_id = item.get("step_id", item.get("criterion_id", item.get("id")))
        if raw_id is None:
            continue
        score = item.get("score", item.get("predicted_score"))
        if isinstance(score, bool) or score not in {0, 1, 2, None}:
            score = None
        result[int(raw_id)] = {**item, "score": score}
    return result


def _infer_candidate(assessment: dict[str, Any], assessment_path: Path) -> str:
    source = assessment.get("metadata", {}).get("source_video")
    if source:
        return Path(str(source)).stem
    config = assessment_path.parent / "client_config.json"
    if config.is_file():
        payload = json.loads(config.read_text(encoding="utf-8"))
        if payload.get("video"):
            return Path(str(payload["video"])).stem
    return assessment_path.parent.name


def _load_long_csv(path: Path, candidate: str | None) -> GoldScores:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"candidate_id", "criterion_id", "score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Gold CSV must contain columns: {sorted(required)}")
    candidates = sorted({str(row["candidate_id"]).strip() for row in rows})
    if not candidate and len(candidates) == 1:
        candidate = candidates[0]
    if candidate not in candidates:
        matches = [value for value in candidates if value in str(candidate) or str(candidate) in value]
        if len(matches) == 1:
            candidate = matches[0]
        else:
            raise ValueError(f"Select --student-name from: {candidates}")
    scores = {
        int(row["criterion_id"]): int(float(row["score"]))
        for row in rows
        if str(row["candidate_id"]).strip() == candidate
    }
    return GoldScores(str(candidate), scores)


def load_gold(
    spec: ExamSpec,
    path: Path,
    candidate: str | None,
    assessment: dict[str, Any],
    assessment_path: Path,
) -> GoldScores:
    gold_format = str(spec.framework.get("gold", {}).get("format", "long_csv"))
    if gold_format == "long_csv":
        return _load_long_csv(path, candidate)
    if gold_format == "aseptic_xlsx":
        labels = legacy_aseptic_gold.load_gold_labels(
            path,
            student_name=candidate,
            assessment=assessment,
            assessment_path=assessment_path,
        )
        return GoldScores(labels.student_name, labels.scores)
    if gold_format == "catheterization_xlsx":
        candidate = candidate or legacy_catheter_gold._infer_student_name(assessment_path)
        candidate = candidate or _infer_candidate(assessment, assessment_path)
        labels = legacy_catheter_gold.load_gold_labels(path, candidate)
        return GoldScores(labels.student_name, labels.scores)
    raise ValueError(
        f"Unsupported gold format {gold_format!r}; use long_csv for a new exam"
    )


def evaluate_assessment(
    *,
    spec: ExamSpec,
    assessment_path: Path,
    workbook: Path | None = None,
    student_name: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    assessment_path = assessment_path.expanduser().resolve()
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    gold_path = (workbook or spec.gold_path())
    if gold_path is None:
        raise ValueError("This rubric has no gold path; pass --gold-workbook")
    gold_path = gold_path.expanduser().resolve()
    gold = load_gold(spec, gold_path, student_name, assessment, assessment_path)
    predictions = _assessment_predictions(assessment)
    rows = []
    for step in spec.rubric["steps"]:
        step_id = step["id"]
        if step_id not in gold.scores:
            raise ValueError(f"Gold label is missing criterion {step_id}")
        prediction = predictions.get(step_id, {})
        rows.append(
            {
                "criterion_id": step_id,
                "description": step["description"],
                "gold_score": gold.scores[step_id],
                "predicted_score": prediction.get("score"),
                "confidence": prediction.get("confidence"),
                "verdict": prediction.get("verdict", "unverifiable"),
                "requires_review": prediction.get("requires_review", True),
            }
        )
    metrics = legacy_catheter_gold.calculate_criterion_metrics(rows)
    result = {
        "schema_version": "1.0",
        "exam_id": spec.exam_id,
        "rubric_id": spec.rubric["rubric_id"],
        "candidate_id": gold.candidate_id,
        "assessment": str(assessment_path),
        "gold_source": str(gold_path),
        "gold_total": sum(gold.scores.values()),
        "predicted_confirmed_total": sum(
            int(row["predicted_score"])
            for row in rows
            if row["predicted_score"] in {0, 1, 2}
        ),
        "maximum_score": spec.rubric["maximum_score"],
        "metrics": metrics,
        "rows": rows,
    }
    output_dir = (output_dir or assessment_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    csv_path = output_dir / "evaluation_steps.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assessment", type=Path)
    parser.add_argument("--exam", required=True)
    parser.add_argument("--knowledge-base", type=Path, action="append")
    parser.add_argument("--gold-workbook", type=Path)
    parser.add_argument("--student-name")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kb = RubricKnowledgeBase(args.knowledge_base or (DEFAULT_KNOWLEDGE_BASE,))
    result = evaluate_assessment(
        spec=kb.get(args.exam),
        assessment_path=args.assessment,
        workbook=args.gold_workbook,
        student_name=args.student_name,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

