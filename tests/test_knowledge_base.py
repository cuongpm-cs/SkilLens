import csv
import json
from pathlib import Path

import pytest

from skillens.evaluate import evaluate_assessment
from skillens.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE,
    RubricKnowledgeBase,
)
from skillens.media import uniform_events


def _minimal_rubric(exam_id: str = "new_exam") -> dict:
    return {
        "framework": {
            "exam_id": exam_id,
            "default_backend": "full_api",
            "gold": {"format": "long_csv", "path": "gold.csv"},
        },
        "title": "Drop-in Exam",
        "maximum_score": 2,
        "steps": [
            {
                "id": 1,
                "description": "Perform the action safely.",
                "allowed_scores": [0, 1, 2],
                "modalities": ["video"],
            }
        ],
    }


def test_bundled_exams_are_discovered_without_python_registry():
    kb = RubricKnowledgeBase((DEFAULT_KNOWLEDGE_BASE,))
    assert [item.exam_id for item in kb.list()] == ["aseptic", "catheterization"]
    assert len(kb.get("medication_preparation").rubric["steps"]) == 30
    assert len(kb.get("urinary_catheterization").rubric["steps"]) == 20


def test_adding_one_rubric_file_registers_a_new_exam(tmp_path: Path):
    path = tmp_path / "drop_in.json"
    path.write_text(json.dumps(_minimal_rubric()), encoding="utf-8")
    kb = RubricKnowledgeBase((tmp_path,))
    spec = kb.get("new_exam")
    assert spec.default_backend == "full_api"
    assert spec.rubric["maximum_score"] == 2


def test_invalid_declared_maximum_is_rejected(tmp_path: Path):
    rubric = _minimal_rubric()
    rubric["maximum_score"] = 99
    (tmp_path / "invalid.json").write_text(json.dumps(rubric), encoding="utf-8")
    with pytest.raises(ValueError, match="Maximum score mismatch"):
        RubricKnowledgeBase((tmp_path,))


def test_generic_long_csv_evaluation_needs_no_exam_code(tmp_path: Path):
    rubric_path = tmp_path / "drop_in.json"
    rubric_path.write_text(json.dumps(_minimal_rubric()), encoding="utf-8")
    gold_path = tmp_path / "gold.csv"
    with gold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "criterion_id", "score"])
        writer.writeheader()
        writer.writerow({"candidate_id": "student_1", "criterion_id": 1, "score": 2})
    assessment_path = tmp_path / "assessment.json"
    assessment_path.write_text(
        json.dumps(
            {
                "metadata": {"source_video": "/data/student_1.mp4"},
                "steps": [{"step_id": 1, "score": 1, "confidence": 0.8}],
            }
        ),
        encoding="utf-8",
    )
    spec = RubricKnowledgeBase((tmp_path,)).get("new_exam")
    result = evaluate_assessment(spec=spec, assessment_path=assessment_path)
    assert result["metrics"]["strict_soft_accuracy"] == 0.5
    assert result["metrics"]["coverage"] == 1.0


def test_uniform_timeline_has_exact_tail_coverage():
    events = uniform_events(65.1234, 30.0)
    assert events[0]["start_seconds"] == 0.0
    assert events[-1]["end_seconds"] == 65.1234
