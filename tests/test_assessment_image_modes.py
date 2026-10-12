from skillens.run import _parser
from skillens.vlm_assessment import (
    _apply_localization_sequence_guard,
    _build_assessment_image_manifest,
    _enforce_visual_evidence_contract,
    _fuse_step_judgments,
    _localization_evidence_timestamps,
    _timeline_timestamps,
)


def _fixture():
    rubric = {
        "rubric_id": "test_rubric",
        "steps": [
            {"id": 1, "allowed_scores": [0, 2], "max_score": 2},
            {"id": 2, "allowed_scores": [0, 1, 2], "max_score": 2},
            {"id": 3, "allowed_scores": [0, 2], "max_score": 2},
        ],
        "critical_rules": [{"id": "critical", "relevant_step_ids": [3]}],
    }
    event_results = [
        {
            "event_id": "event_000",
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "observations": [
                {
                    "observation_id": "obs_1",
                    "step_id": 1,
                    "status": "complete",
                    "confidence": 0.9,
                    "evidence_seconds": [10.0],
                    "description": "This prose must not enter the image manifest.",
                },
                {
                    "observation_id": "obs_3",
                    "step_id": 3,
                    "status": "complete",
                    "confidence": 0.8,
                    "evidence_seconds": [30.0, 35.0, 40.0],
                    "description": "Another observation interpretation.",
                },
            ],
        }
    ]
    assessment = {
        "steps": [
            {"step_id": 1, "score": 2, "confidence": 0.95, "evidence": []},
            {
                "step_id": 2,
                "score": None,
                "confidence": 0.2,
                "requires_review": True,
                "evidence": [],
            },
            {"step_id": 3, "score": 2, "confidence": 0.7, "evidence": []},
        ]
    }
    return rubric, event_results, assessment


def _manifest(mode: str):
    rubric, events, assessment = _fixture()
    return _build_assessment_image_manifest(
        mode=mode,
        rubric=rubric,
        event_results=events,
        preliminary_assessment=assessment,
        uncertainty_threshold=0.75,
        images_per_criterion=3,
        context_frames=1,
        critical_criterion_images=6,
        rescue_step_limit=30,
    )


def test_default_cli_uses_text_only_assessment_mode():
    args = _parser().parse_args([])
    assert args.observation_provider == "openai-compatible"
    assert args.assessment_provider == "openai-compatible"
    assert args.assessment_images == "never"
    assert args.assessment_image_confidence_threshold == 0.75
    assert args.assessment_visual_override_confidence == 0.85
    assert args.assessment_visual_override_margin == 0.15
    assert args.assessment_images_per_criterion == 3
    assert args.assessment_context_frames == 1
    assert args.critical_criterion_images == 6
    assert args.criterion_localization == "off"
    assert args.localization_candidates == 3
    assert args.localization_coarse_seconds == 5.0
    assert args.localization_search_radius == 5.0
    assert args.localization_refinement_fps == 8.0
    assert args.localization_evidence_images == 12
    assert args.localization_confidence_threshold == 0.70
    assert args.localization_workers == 4
    assert args.yolo_execution == "local"
    assert args.modal_yolo_app == "skillens-yolo-perception"
    assert args.modal_yolo_volume == "skillens-yolo-data"


def test_never_mode_selects_no_images():
    manifest = _manifest("never")
    assert manifest["targets"] == []
    assert manifest["summary"]["selected_images"] == 0


def test_rescue_mode_selects_only_unresolved_criterion_with_fallback():
    manifest = _manifest("rescue")
    assert [item["step_id"] for item in manifest["targets"]] == [2]
    assert manifest["targets"][0]["selection_basis"] == "temporal_prior"


def test_uncertain_mode_selects_unresolved_and_low_confidence_criteria():
    manifest = _manifest("uncertain")
    assert [item["step_id"] for item in manifest["targets"]] == [2, 3]
    assert manifest["targets"][0]["selection_basis"] == "temporal_prior"
    assert manifest["targets"][1]["selection_basis"] == "observation_timestamp"
    assert manifest["selection_policy"]["uncertainty_threshold"] == 0.75


def test_evidence_mode_uses_only_direct_evidence_timestamps():
    manifest = _manifest("evidence")
    assert [item["step_id"] for item in manifest["targets"]] == [1, 3]
    assert manifest["targets"][0]["timestamps"] == [9.0, 10.0, 11.0]
    assert all(
        item["selection_basis"] == "observation_timestamp"
        for item in manifest["targets"]
    )
    assert "description" not in str(manifest)
    assert len(manifest["targets"][1]["timestamps"]) > 3
    assert len(manifest["targets"][0]["timestamps"]) <= 3
    assert len(manifest["targets"][1]["timestamps"]) <= 6


def test_always_mode_covers_every_criterion():
    manifest = _manifest("always")
    assert [item["step_id"] for item in manifest["targets"]] == [1, 2, 3]
    step_two = next(item for item in manifest["targets"] if item["step_id"] == 2)
    assert step_two["selection_basis"] == "temporal_prior"


def _fuse(text_score, text_confidence, visual_score, visual_confidence, seconds):
    verdicts = {0: "incorrect", 1: "partial", 2: "complete", None: "unverifiable"}
    text = {
        "steps": [
            {
                "step_id": 1,
                "score": text_score,
                "verdict": verdicts[text_score],
                "confidence": text_confidence,
                "evidence_ids": ["obs_1"],
            }
        ]
    }
    preliminary = {
        "steps": [
            {
                "step_id": 1,
                "score": text_score,
                "confidence": text_confidence,
                "requires_review": text_score is None,
            }
        ]
    }
    visual = {
        "steps": [
            {
                "step_id": 1,
                "score": visual_score,
                "verdict": verdicts[visual_score],
                "confidence": visual_confidence,
                "evidence_seconds": seconds,
                "reason": "visual reason",
            }
        ]
    }
    rubric = {"steps": [{"id": 1, "allowed_scores": [0, 1, 2]}]}
    return _fuse_step_judgments(
        text,
        visual,
        preliminary,
        rubric,
        visual_override_confidence=0.85,
        visual_override_margin=0.15,
    )["steps"][0]


def test_visual_disagreement_does_not_replace_confident_text():
    item = _fuse(2, 0.9, 1, 0.7, [10.0])
    assert item["score"] == 2
    assert item["fusion_decision"] == "text_retained_visual_disagreement"


def test_visual_evidence_can_rescue_unresolved_text():
    item = _fuse(None, 0.2, 1, 0.8, [10.0])
    assert item["score"] == 1
    assert item["fusion_decision"] == "visual_rescue"


def test_strong_visual_evidence_can_override_weaker_text():
    item = _fuse(1, 0.6, 2, 0.95, [10.0])
    assert item["score"] == 2
    assert item["fusion_decision"] == "visual_override_strong_evidence"


def test_visual_result_without_direct_evidence_does_not_override_text():
    item = _fuse(1, 0.6, 2, 0.99, [])
    assert item["score"] == 1
    assert item["fusion_decision"] == "text_retained_visual_insufficient"


def test_visual_review_can_rescue_a_step_missing_from_text_response():
    result = _fuse_step_judgments(
        {"steps": []},
        {
            "steps": [
                {
                    "step_id": 1,
                    "score": 2,
                    "verdict": "complete",
                    "confidence": 0.9,
                    "evidence_seconds": [10.0],
                }
            ]
        },
        {"steps": [{"step_id": 1, "score": None, "confidence": 0.0}]},
        {"steps": [{"id": 1, "allowed_scores": [0, 2]}]},
        visual_override_confidence=0.85,
        visual_override_margin=0.15,
    )
    assert result["steps"][0]["score"] == 2
    assert result["steps"][0]["fusion_decision"] == "visual_rescue"


def test_localization_samples_full_timeline_at_interval_midpoints():
    assert _timeline_timestamps(20.0, 5.0) == [2.5, 7.5, 12.5, 17.5]


def test_localization_builds_before_during_after_evidence_sequence():
    timestamps = _localization_evidence_timestamps(
        start=10.0,
        end=11.0,
        best=10.5,
        duration=20.0,
        count=12,
    )
    assert len(timestamps) == 12
    assert timestamps == sorted(timestamps)
    assert timestamps[:4] == [8.0, 8.625, 9.25, 9.875]
    assert timestamps[4:8] == [10.0, 10.333, 10.667, 11.0]
    assert timestamps[-4:] == [11.125, 11.75, 12.375, 13.0]


def test_sequence_guard_rejects_repeated_action_from_wrong_phase():
    timestamps = [82.0, 172.0, 550.0, 284.0, 305.0, 409.0, 448.0, 315.0, 498.0]
    targets = []
    for step_id, timestamp in enumerate(timestamps, start=1):
        targets.append(
            {
                "step_id": step_id,
                "timestamps": [timestamp],
                "localization": {
                    "status": "verified",
                    "best_timestamp": timestamp,
                    "confidence": 0.9,
                    "reason": "visible",
                },
            }
        )
    manifest = {
        "targets": targets,
        "selection_policy": {"criterion_localization": {}},
        "summary": {},
    }
    _apply_localization_sequence_guard(manifest, tolerance_seconds=5.0)
    assert manifest["summary"]["localization_sequence_rejected_step_ids"] == [3, 8]
    assert targets[2]["timestamps"] == []
    assert targets[7]["timestamps"] == []
    assert targets[4]["timestamps"] == [305.0]


def test_missing_action_cannot_be_converted_to_zero_score():
    result = _enforce_visual_evidence_contract(
        {
            "criterion_action_visible": False,
            "direct_error_visible": False,
            "score": 0,
            "verdict": "incorrect",
            "confidence": 1.0,
            "evidence_seconds": [10.0],
            "reason": "The requested action was not shown.",
        }
    )
    assert result["score"] is None
    assert result["verdict"] == "unverifiable"
    assert result["evidence_seconds"] == []
    assert result["evidence_contract_adjusted"] is True


def test_directly_visible_error_can_keep_zero_score():
    result = _enforce_visual_evidence_contract(
        {
            "criterion_action_visible": True,
            "direct_error_visible": True,
            "score": 0,
            "verdict": "incorrect",
            "confidence": 0.9,
            "evidence_seconds": [10.0],
        }
    )
    assert result["score"] == 0
    assert result["evidence_seconds"] == [10.0]
    assert result["evidence_contract_adjusted"] is False
