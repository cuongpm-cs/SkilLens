"""The public defaults must be usable with two independent API endpoints."""

from pathlib import Path

import pytest

from skillens.model_config import (
    DEFAULT_ASSESSMENT_MODEL,
    DEFAULT_OBSERVATION_MODEL,
    is_text_only_assessment_model,
    resolve_model_endpoint,
)
from skillens.run import main


def test_default_compatible_models_are_stage_specific() -> None:
    env = {
        "OBSERVATION_BASE_URL": "http://localhost:8001/v1",
        "OBSERVATION_API_KEY": "local",
        "ASSESSMENT_BASE_URL": "http://localhost:8002/v1",
        "ASSESSMENT_API_KEY": "local",
    }
    observation = resolve_model_endpoint(
        stage="observation",
        provider="openai-compatible",
        model=None,
        base_url=None,
        environment=env,
        default_openai_model="gpt-4.1",
        default_compatible_model=DEFAULT_OBSERVATION_MODEL,
    )
    assessment = resolve_model_endpoint(
        stage="assessment",
        provider="openai-compatible",
        model=None,
        base_url=None,
        environment=env,
        default_openai_model="gpt-4.1",
        default_compatible_model=DEFAULT_ASSESSMENT_MODEL,
    )
    assert observation.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert observation.base_url == "http://localhost:8001/v1"
    assert assessment.model == "gpt-oss-20b"
    assert assessment.base_url == "http://localhost:8002/v1"


@pytest.mark.parametrize("model", ["gpt-oss-20b", "openai/gpt-oss-20b", "gpt-oss-120b"])
def test_gpt_oss_is_known_text_only(model: str) -> None:
    assert is_text_only_assessment_model(model)
    assert not is_text_only_assessment_model("Qwen/Qwen3-VL-8B-Instruct")


def test_incompatible_image_mode_fails_before_video_processing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OBSERVATION_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("OBSERVATION_API_KEY", "local")
    monkeypatch.setenv("ASSESSMENT_BASE_URL", "http://localhost:8002/v1")
    monkeypatch.setenv("ASSESSMENT_API_KEY", "local")
    with pytest.raises(ValueError, match="cannot review images"):
        main([
            str(tmp_path / "not-a-video.mp4"),
            "--exam", "catheterization",
            "--assessment-images", "uncertain",
        ])
