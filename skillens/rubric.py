"""Small rubric helpers shared by optional pipeline stages."""

from __future__ import annotations

from typing import Any


def rubric_requires_audio(rubric: dict[str, Any]) -> bool:
    return any("audio" in step.get("modalities", []) for step in rubric["steps"])

