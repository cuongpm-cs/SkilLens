import json
from pathlib import Path

from skillens.vlm_assessment import _assessment_rubric, _observation_rubric


ROOT = Path(__file__).resolve().parents[1]


def test_observation_view_is_compact_and_evidence_focused():
    rubric = json.loads(
        (ROOT / "knowledge_base/rubrics/catheterization.json").read_text(encoding="utf-8")
    )
    view = _observation_rubric(rubric)
    assert len(view) == 20
    assert "observable_evidence" in view[0]
    assert "scoring_rules" not in view[0]
    assert "critical_rules" not in view[0]


def test_assessment_view_contains_full_scoring_context_but_no_assets():
    rubric = json.loads(
        (ROOT / "knowledge_base/rubrics/catheterization.json").read_text(encoding="utf-8")
    )
    view = _assessment_rubric(rubric)
    assert "assessment_policy" in view
    assert "sequence_constraints" in view
    assert "critical_rules" in view
    assert "score_calculation" in view
    assert "scoring_rules" in view["steps"][0]
    assert "framework" not in view
    assert "assets" not in json.dumps(view)
