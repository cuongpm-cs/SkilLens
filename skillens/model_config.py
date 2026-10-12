"""Model endpoint configuration shared by both inference stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


PROVIDER_OPENAI = "openai"
PROVIDER_COMPATIBLE = "openai-compatible"
DEFAULT_OBSERVATION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ASSESSMENT_MODEL = "gpt-oss-20b"
PROVIDER_ALIASES = {
    "openai": PROVIDER_OPENAI,
    "compatible": PROVIDER_COMPATIBLE,
    "modal": PROVIDER_COMPATIBLE,
    "openai-compatible": PROVIDER_COMPATIBLE,
    "openai_compatible": PROVIDER_COMPATIBLE,
}


@dataclass(frozen=True)
class ModelEndpoint:
    """Resolved API configuration for one pipeline stage."""

    stage: str
    provider: str
    model: str
    api_key: str
    base_url: str | None


def normalize_provider(value: str) -> str:
    """Normalize a public or legacy provider name."""
    normalized = PROVIDER_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        supported = ", ".join(sorted({PROVIDER_OPENAI, PROVIDER_COMPATIBLE}))
        raise ValueError(f"Unsupported provider '{value}'. Choose one of: {supported}.")
    return normalized


def is_text_only_assessment_model(model: str) -> bool:
    """Reject known GPT-OSS aliases from image-review modes before inference."""
    return str(model).strip().casefold().split("/")[-1] in {
        "gpt-oss-20b",
        "gpt-oss-120b",
    }


def _first_value(
    explicit: str | None,
    environment: Mapping[str, str],
    names: tuple[str, ...],
) -> str | None:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for name in names:
        value = environment.get(name)
        if value and value.strip():
            return value.strip()
    return None


def resolve_model_endpoint(
    *,
    stage: str,
    provider: str,
    model: str | None,
    base_url: str | None,
    environment: Mapping[str, str] | None = None,
    default_openai_model: str,
    default_compatible_model: str | None = None,
) -> ModelEndpoint:
    """Resolve one stage without leaking one stage's settings into the other."""
    if stage not in {"observation", "assessment"}:
        raise ValueError("stage must be 'observation' or 'assessment'.")
    env = os.environ if environment is None else environment
    requested_provider = str(provider).strip().lower()
    provider = normalize_provider(provider)
    prefix = stage.upper()

    if provider == PROVIDER_OPENAI:
        resolved_model = _first_value(
            model,
            env,
            (f"{prefix}_MODEL", "OPENAI_MODEL"),
        ) or default_openai_model
        api_key = _first_value(None, env, ("OPENAI_API_KEY",))
        if not api_key:
            raise ValueError(
                f"{stage.capitalize()} provider 'openai' requires OPENAI_API_KEY."
            )
        return ModelEndpoint(
            stage=stage,
            provider=provider,
            model=resolved_model,
            api_key=api_key,
            base_url=None,
        )

    is_modal = requested_provider == "modal"
    if stage == "observation":
        legacy_model_names = ("VLM_MODEL", "QWEN3_VL_MODEL")
        legacy_base_url_names = ("VLM_BASE_URL", "QWEN3_VL_BASE_URL")
        legacy_api_key_names = ("VLM_API_KEY", "VLLM_API_KEY")
    else:
        legacy_model_names = ("LLM_MODEL",)
        legacy_base_url_names = ("LLM_BASE_URL",)
        legacy_api_key_names = ("LLM_API_KEY",)
    if is_modal:
        legacy_model_names = ()
        legacy_base_url_names = ()
        legacy_api_key_names = ()
    resolved_model = _first_value(
        model,
        env,
        (
            f"{prefix}_MODEL",
            f"MODAL_{prefix}_MODEL",
            "MODAL_MODEL",
            *legacy_model_names,
        ),
    ) or default_compatible_model
    resolved_base_url = _first_value(
        base_url,
        env,
        (
            f"{prefix}_BASE_URL",
            f"MODAL_{prefix}_BASE_URL",
            "MODAL_BASE_URL",
            "OPENAI_COMPATIBLE_BASE_URL",
            *legacy_base_url_names,
        ),
    )
    api_key = _first_value(
        None,
        env,
        (
            f"{prefix}_API_KEY",
            f"MODAL_{prefix}_API_KEY",
            "MODAL_API_KEY",
            "OPENAI_COMPATIBLE_API_KEY",
            *legacy_api_key_names,
        ),
    )
    missing = []
    if not resolved_model:
        missing.append(f"{prefix}_MODEL or --{stage}-model")
    if not resolved_base_url:
        missing.append(f"{prefix}_BASE_URL or --{stage}-base-url")
    if not api_key:
        missing.append(f"{prefix}_API_KEY")
    if missing:
        provider_label = "modal" if is_modal else "openai-compatible"
        raise ValueError(
            f"{stage.capitalize()} provider '{provider_label}' is missing: "
            + ", ".join(missing)
            + ". For an unauthenticated local server, set the API key to any "
            "non-empty value such as 'local'."
        )
    return ModelEndpoint(
        stage=stage,
        provider=provider,
        model=resolved_model,
        api_key=api_key,
        base_url=resolved_base_url,
    )
