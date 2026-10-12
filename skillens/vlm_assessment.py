"""Evidence-grounded observation and multimodal rubric assessment."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import cv2
import numpy as np

from .branding import FRAMEWORK_FULL_NAME, FRAMEWORK_NAME
from openai import BadRequestError, OpenAI

from .model_config import PROVIDER_OPENAI, normalize_provider


VALID_OBSERVATION_STATUS = {"complete", "partial", "incorrect", "unclear"}
SCORE_BY_VERDICT = {"complete": 2, "partial": 1, "incorrect": 0, "unverifiable": None}
MAX_LOCAL_JSON_REPAIRS = 32
ASSESSMENT_IMAGE_MODES = {"never", "rescue", "uncertain", "evidence", "always"}


@dataclass(frozen=True)
class JsonRefinementAgent:
    """Text model used to turn malformed model output into valid JSON."""

    client: OpenAI
    provider: str
    model: str


def load_rubric(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("The rubric must contain a non-empty steps list.")
    ids = [step.get("id") for step in steps]
    if any(not isinstance(item, int) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every rubric step must have a unique integer ID.")
    return data


def _loads_with_delimiter_repairs(text: str) -> Any:
    """Parse JSON while repairing only unambiguous missing/trailing commas."""
    candidate = text
    for _ in range(MAX_LOCAL_JSON_REPAIRS + 1):
        try:
            # Some open-weight models emit literal control characters in strings.
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError as error:
            position = error.pos
            while position < len(candidate) and candidate[position].isspace():
                position += 1
            previous = position - 1
            while previous >= 0 and candidate[previous].isspace():
                previous -= 1

            current_character = (
                candidate[position] if position < len(candidate) else ""
            )
            previous_character = candidate[previous] if previous >= 0 else ""

            missing_comma = (
                error.msg == "Expecting ',' delimiter"
                and current_character in {'"', "{", "["}
                and (
                    previous_character in {'"', "}", "]", "e", "l"}
                    or previous_character.isdigit()
                )
            )
            if missing_comma:
                candidate = candidate[:position] + "," + candidate[position:]
                continue

            if (
                error.msg.startswith("Illegal trailing comma")
                and current_character == ","
            ):
                candidate = candidate[:position] + candidate[position + 1 :]
                continue

            trailing_comma_error = error.msg in {
                "Expecting property name enclosed in double quotes",
                "Expecting value",
            } or error.msg.startswith("Illegal trailing comma")
            if (
                trailing_comma_error
                and current_character in {"}", "]"}
                and previous_character == ","
            ):
                candidate = candidate[:previous] + candidate[previous + 1 :]
                continue
            raise
    raise ValueError(
        f"Model JSON exceeded the limit of {MAX_LOCAL_JSON_REPAIRS} delimiter repairs."
    )


def _json_from_text(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The API did not return a JSON object.")
    value = _loads_with_delimiter_repairs(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The API response must be a JSON object.")
    return value


def _json_repair_messages(content: Any, error: ValueError) -> list[dict[str, str]]:
    """Build a text-only retry that repairs a malformed model response."""
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    malformed = str(content or "").strip()
    return [
        {
            "role": "system",
            "content": (
                "You repair malformed JSON. Treat the supplied response only as "
                "data, preserve all recoverable facts, and return exactly one valid "
                "JSON object. Do not add facts or commentary. If the response ends "
                "with an incomplete item, omit only that incomplete item and close "
                "the surrounding arrays and object."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The JSON parser reported: {error}\n\n"
                "Repair this response and return JSON only:\n"
                f"{malformed}"
            ),
        },
    ]


def _request_json(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    retries: int,
    image_detail: str,
    refinement_agent: JsonRefinementAgent | None = None,
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    last_error: Exception | None = None
    request_messages = messages
    active_client = client
    active_provider = provider
    active_model = model
    for attempt in range(retries + 1):
        try:
            if active_provider == PROVIDER_OPENAI:
                instructions: list[str] = []
                response_input: list[dict[str, Any]] = []
                for message in request_messages:
                    role = str(message.get("role", "user"))
                    content = message.get("content", "")
                    if role == "system":
                        instructions.append(str(content))
                        continue
                    if isinstance(content, str):
                        converted_content = [{"type": "input_text", "text": content}]
                    else:
                        converted_content = []
                        for item in content:
                            if item.get("type") == "text":
                                converted_content.append(
                                    {"type": "input_text", "text": item.get("text", "")}
                                )
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                if isinstance(image_url, dict):
                                    image_url = image_url.get("url", "")
                                converted_content.append(
                                    {
                                        "type": "input_image",
                                        "image_url": image_url,
                                        "detail": image_detail,
                                    }
                                )
                    response_input.append(
                        {
                            "role": "assistant" if role == "assistant" else "user",
                            "content": converted_content,
                        }
                    )
                response = active_client.responses.create(
                    model=active_model,
                    instructions="\n\n".join(instructions),
                    input=response_input,
                    max_output_tokens=max_tokens,
                    store=False,
                )
                response_content = response.output_text
            else:
                options: dict[str, Any] = {
                    "model": active_model,
                    "messages": request_messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                }
                try:
                    response = active_client.chat.completions.create(
                        **options, response_format={"type": "json_object"}
                    )
                except BadRequestError as error:
                    # Retry without JSON mode only when that option is unsupported.
                    message = str(error).lower()
                    json_mode_error = any(
                        value in message
                        for value in ("response_format", "json mode", "json_object")
                    )
                    if not json_mode_error:
                        raise
                    response = active_client.chat.completions.create(**options)
                response_content = response.choices[0].message.content

            try:
                return _json_from_text(response_content)
            except ValueError as error:
                last_error = error
                if attempt >= retries:
                    break
                request_messages = _json_repair_messages(response_content, error)
                if refinement_agent is not None:
                    active_client = refinement_agent.client
                    active_provider = normalize_provider(refinement_agent.provider)
                    active_model = refinement_agent.model
                    print(
                        "Model returned malformed JSON; sending it to the "
                        f"refinement agent ({active_model}, "
                        f"attempt {attempt + 2}/{retries + 1})."
                    )
                else:
                    print(
                        "Model returned malformed JSON; retrying with a text-only "
                        f"repair request ({attempt + 2}/{retries + 1})."
                    )
        except BadRequestError as error:
            raise RuntimeError(f"Model API rejected the request: {error}") from error
        except Exception as error:  # Network errors and malformed model JSON are retryable.
            last_error = error
            if attempt >= retries:
                break
            time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"Model API failed after {retries + 1} attempts: {last_error}")


def _sample_timestamps(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("The number of frames per event must be positive.")
    span = max(0.001, end - start)
    return [round(start + span * (index + 0.5) / count, 3) for index in range(count)]


def _read_frame(capture: cv2.VideoCapture, timestamp: float, max_side: int) -> bytes:
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read the frame at {timestamp:.3f}s")
    height, width = frame.shape[:2]
    largest = max(width, height)
    if largest > max_side:
        scale = max_side / largest
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("Cannot encode the frame as JPEG.")
    return encoded.tobytes()


def _image_item(jpeg: bytes) -> dict[str, Any]:
    payload = base64.b64encode(jpeg).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
    }


def _observation_rubric(rubric: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only fields that help the VLM find direct visible evidence."""
    return [
        {
            "id": step["id"],
            "key": step.get("key"),
            "description": step.get("description"),
            "required_objects": step.get("required_objects", []),
            "modalities": step.get("modalities", ["video"]),
            "observable_evidence": step.get("observable_evidence", []),
        }
        for step in rubric["steps"]
    ]


def _assessment_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Expose complete scoring context without leaking framework credentials/assets."""
    global_fields = (
        "rubric_id",
        "title",
        "scope_note",
        "interpretation_note",
        "maximum_score",
        "score_scale",
        "assessment_policy",
        "score_calculation",
        "phases",
        "sequence_constraints",
        "critical_rules",
    )
    step_fields = (
        "id",
        "key",
        "description",
        "phase",
        "assessment_scope",
        "allowed_scores",
        "max_score",
        "modalities",
        "required_objects",
        "observable_evidence",
        "scoring_rules",
        "do_not_infer",
        "critical_rule_reference",
    )
    result = {key: rubric[key] for key in global_fields if key in rubric}
    result["steps"] = [
        {key: step[key] for key in step_fields if key in step}
        for step in rubric["steps"]
    ]
    return result


def _analyze_event(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    raw_video: Path,
    annotated_video: Path | None,
    event: dict[str, Any],
    rubric: dict[str, Any],
    frames_per_event: int,
    max_image_side: int,
    include_annotated_frames: bool,
    max_tokens: int,
    retries: int,
    image_detail: str,
    refinement_agent: JsonRefinementAgent | None,
) -> dict[str, Any]:
    timestamps = _sample_timestamps(
        float(event["start_seconds"]), float(event["end_seconds"]), frames_per_event
    )
    prompt = f"""
You are the OBSERVATION stage of a nursing-skills video assessment system. Describe
only direct visual evidence in the supplied frames. Do not grade the full procedure
and do not infer actions that may have occurred between frames. This is
{event['event_id']} from {event['start_seconds']:.3f}s to
{event['end_seconds']:.3f}s.

{len(rubric['steps'])}-criterion examination rubric:
{json.dumps(_observation_rubric(rubric), ensure_ascii=False)}

Perception/tracking evidence from the selected backend (a fallible hint, not ground truth):
{json.dumps(event, ensure_ascii=False)}

Return one observation for each action supported by direct evidence. Every value in
evidence_seconds must be one of the supplied frame timestamps. Use status "unclear"
when the action is not sufficiently visible. A missing object detection is not evidence
that an action did not occur.

Return JSON only:
{{
  "observations": [
    {{
      "step_id": 1,
      "status": "complete|partial|incorrect|unclear",
      "confidence": 0.0,
      "evidence_seconds": [0.0],
      "description": "brief description of the direct visual evidence"
    }}
  ],
  "visual_limitations": [],
  "safety_concerns": []
}}
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    raw_capture = cv2.VideoCapture(str(raw_video))
    if not raw_capture.isOpened():
        raise RuntimeError(f"Cannot open the raw video: {raw_video}")
    annotated_capture: cv2.VideoCapture | None = None
    if include_annotated_frames and annotated_video is not None:
        annotated_capture = cv2.VideoCapture(str(annotated_video))
        if not annotated_capture.isOpened():
            raw_capture.release()
            raise RuntimeError(f"Cannot open the annotated video: {annotated_video}")
    try:
        for timestamp in timestamps:
            content.append(
                {"type": "text", "text": f"Raw frame at {timestamp:.3f} seconds"}
            )
            content.append(_image_item(_read_frame(raw_capture, timestamp, max_image_side)))
            if annotated_capture is not None:
                content.append(
                    {
                        "type": "text",
                        "text": f"YOLO/hand reference at {timestamp:.3f} seconds",
                    }
                )
                content.append(
                    _image_item(_read_frame(annotated_capture, timestamp, max_image_side))
                )
    finally:
        raw_capture.release()
        if annotated_capture is not None:
            annotated_capture.release()

    raw = _request_json(
        client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative visual-evidence extractor. Return JSON "
                    "only and never invent details that are not visible."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        retries=retries,
        image_detail=image_detail,
        refinement_agent=refinement_agent,
    )
    valid_step_ids = {step["id"] for step in rubric["steps"]}
    normalized: list[dict[str, Any]] = []
    for item in raw.get("observations", []):
        if not isinstance(item, dict) or item.get("step_id") not in valid_step_ids:
            continue
        status = str(item.get("status", "unclear"))
        if status not in VALID_OBSERVATION_STATUS:
            status = "unclear"
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_seconds = []
        for value in item.get("evidence_seconds", []):
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            nearest = min(timestamps, key=lambda timestamp: abs(timestamp - value_float))
            if abs(nearest - value_float) <= 0.25:
                evidence_seconds.append(nearest)
        evidence_seconds = sorted(set(evidence_seconds))
        normalized.append(
            {
                "observation_id": f"{event['event_id']}_obs_{len(normalized):03d}",
                "event_id": event["event_id"],
                "step_id": item["step_id"],
                "status": status,
                "confidence": round(confidence, 4),
                "evidence_seconds": evidence_seconds,
                "description": str(item.get("description", "")).strip(),
            }
        )
    return {
        "event_id": event["event_id"],
        "start_seconds": event["start_seconds"],
        "end_seconds": event["end_seconds"],
        "sampled_timestamps": timestamps,
        "objects": event.get("objects", []),
        "observations": normalized,
        "visual_limitations": [
            str(value) for value in raw.get("visual_limitations", []) if str(value)
        ],
        "safety_concerns": [
            str(value) for value in raw.get("safety_concerns", []) if str(value)
        ],
    }


def _judge(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    rubric: dict[str, Any],
    event_results: list[dict[str, Any]],
    max_tokens: int,
    retries: int,
    image_detail: str,
    audio_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observations = [
        observation
        for result in event_results
        for observation in result["observations"]
    ]
    prompt = f"""
You are the EVIDENCE-TO-RUBRIC stage for the complete video timeline.

Complete assessment rubric and scoring policy:
{json.dumps(_assessment_rubric(rubric), ensure_ascii=False)}

Observations:
{json.dumps(observations, ensure_ascii=False)}

Audio evidence (may be disabled, unavailable, or speaker-unverified):
{json.dumps(audio_evidence or {"enabled": False}, ensure_ascii=False)}

Mandatory rules:
- Use only observation_id values present in the supplied list.
- Use only audio_id values present in supplied audio evidence and only for a
  criterion whose modalities include audio.
- Follow allowed_scores for each criterion. Use partial=1 only when 1 is legal;
  never invent an illegal score.
- Apply each criterion's scoring_rules, do_not_infer rules, and assessment_policy.
- Check sequence_constraints and critical_rules across the complete timeline.
- Do not invent a deduction. When a critical rule requires continuous evidence or
  human confirmation, report it in safety_concerns for review rather than applying
  an unsupported automatic deduction.
- Missing or insufficient evidence must be unverifiable with score=null. Never turn
  missing evidence into a score of zero.
- Perception, ASR, and VLM evidence can be wrong. Treat low-confidence evidence conservatively.
- Missing or speaker-unverified audio is not proof that the candidate omitted speech.
- Return exactly one result for every step_id.

Return JSON only:
{{
  "steps": [
    {{
      "step_id": 1,
      "score": 2,
      "verdict": "complete|partial|incorrect|unverifiable",
      "confidence": 0.0,
      "evidence_ids": ["event_000_obs_000"],
      "audio_evidence_ids": ["audio_0000"],
      "reason": "brief rationale"
    }}
  ],
  "overall_feedback": "...",
  "safety_concerns": []
}}
""".strip()
    return _request_json(
        client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a conservative assessor. Insufficient evidence must be "
                    "unverifiable. Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        retries=retries,
        image_detail=image_detail,
    )


def _candidate_timestamp_for_step(
    step_id: int,
    assessment: dict[str, Any],
    event_results: list[dict[str, Any]],
    total_steps: int,
) -> float:
    """Choose a targeted frame using direct hints, then temporal interpolation."""
    direct_candidates: list[tuple[float, float]] = []
    for event in event_results:
        for observation in event["observations"]:
            if observation.get("step_id") != step_id:
                continue
            confidence = float(observation.get("confidence", 0.0) or 0.0)
            status = str(observation.get("status", "unclear"))
            if confidence <= 0.0 or status == "unclear":
                continue
            for timestamp in observation.get("evidence_seconds", []):
                direct_candidates.append((confidence, float(timestamp)))
    if direct_candidates:
        return round(max(direct_candidates)[1], 3)

    known: list[tuple[int, float]] = []
    for result in assessment["steps"]:
        timestamps = [
            float(timestamp)
            for evidence in result.get("evidence", [])
            for timestamp in evidence.get("evidence_seconds", [])
        ]
        if result.get("score") is not None and timestamps:
            known.append((int(result["step_id"]), min(timestamps)))

    previous = max(
        (item for item in known if item[0] < step_id),
        default=None,
        key=lambda item: item[0],
    )
    following = min(
        (item for item in known if item[0] > step_id),
        default=None,
        key=lambda item: item[0],
    )
    duration = max(float(event["end_seconds"]) for event in event_results)
    if previous and following:
        if previous[1] < following[1]:
            ratio = (step_id - previous[0]) / (following[0] - previous[0])
            return round(previous[1] + ratio * (following[1] - previous[1]), 3)
        # Conflicting temporal anchors are less useful than the rubric-position prior.
        return round(duration * step_id / (total_steps + 1), 3)
    if previous and previous[1] < duration:
        ratio = (step_id - previous[0]) / max(1, total_steps + 1 - previous[0])
        return round(previous[1] + ratio * (duration - previous[1]), 3)
    if following and following[1] > 0:
        ratio = step_id / following[0]
        return round(ratio * following[1], 3)
    return round(duration * step_id / (total_steps + 1), 3)


def _critical_step_ids(rubric: dict[str, Any]) -> set[int]:
    """Return criteria explicitly connected to a critical rubric rule."""
    result: set[int] = set()
    for rule in rubric.get("critical_rules") or []:
        for value in rule.get("relevant_step_ids", []):
            if isinstance(value, int):
                result.add(value)
    for step in rubric["steps"]:
        if step.get("critical_rule_reference") or step.get("critical_rule_references"):
            result.add(int(step["id"]))
    return result


def _direct_timestamps_for_step(
    step_id: int, event_results: list[dict[str, Any]]
) -> list[float]:
    """Rank direct observation timestamps without exposing observation prose."""
    candidates: dict[float, float] = {}
    for event in event_results:
        for observation in event["observations"]:
            if observation.get("step_id") != step_id:
                continue
            confidence = float(observation.get("confidence", 0.0) or 0.0)
            if confidence <= 0.0 or observation.get("status") == "unclear":
                continue
            for raw_timestamp in observation.get("evidence_seconds", []):
                try:
                    timestamp = round(float(raw_timestamp), 3)
                except (TypeError, ValueError):
                    continue
                candidates[timestamp] = max(confidence, candidates.get(timestamp, 0.0))
    return [
        timestamp
        for timestamp, _ in sorted(
            candidates.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _select_assessment_timestamps(
    anchors: list[float],
    *,
    duration: float,
    max_images: int,
    context_frames: int,
) -> list[float]:
    """Select diverse anchors, then add one-second temporal context around them."""
    unique_anchors: list[float] = []
    for value in anchors:
        timestamp = min(duration, max(0.0, round(float(value), 3)))
        if all(abs(timestamp - existing) > 0.25 for existing in unique_anchors):
            unique_anchors.append(timestamp)
    if not unique_anchors:
        return []

    if len(unique_anchors) > max_images:
        if max_images == 1:
            selected = [unique_anchors[0]]
        else:
            chronological = sorted(unique_anchors)
            selected = [
                chronological[round(index * (len(chronological) - 1) / (max_images - 1))]
                for index in range(max_images)
            ]
    else:
        selected = list(unique_anchors)

    if len(selected) >= max_images:
        return sorted(selected[:max_images])

    for anchor in unique_anchors:
        for distance in range(1, context_frames + 1):
            for candidate in (anchor - distance, anchor + distance):
                candidate = min(duration, max(0.0, round(candidate, 3)))
                if all(abs(candidate - existing) > 0.25 for existing in selected):
                    selected.append(candidate)
                if len(selected) >= max_images:
                    return sorted(selected)
    return sorted(selected[:max_images])


def _timeline_timestamps(duration: float, interval_seconds: float) -> list[float]:
    """Sample the complete timeline at interval midpoints for coarse search."""
    if interval_seconds <= 0:
        raise ValueError("The timeline interval must be positive.")
    if duration <= 0:
        return []
    values: list[float] = []
    timestamp = min(duration, interval_seconds / 2.0)
    while timestamp < duration:
        values.append(round(timestamp, 3))
        timestamp += interval_seconds
    if not values:
        values.append(round(duration / 2.0, 3))
    return values


def _contact_sheets(
    raw_video: Path,
    timestamps: list[float],
    *,
    max_side: int,
    prefix: str,
    frames_per_sheet: int = 40,
    columns: int = 5,
) -> list[tuple[list[float], bytes]]:
    """Render timestamped frame grids so a VLM can scan long intervals cheaply."""
    if not timestamps:
        return []
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open the raw video: {raw_video}")
    sheet_width = max(640, min(1600, int(max_side)))
    tile_width = max(1, sheet_width // columns)
    tile_height = max(1, round(tile_width * 9 / 16))
    sheets: list[tuple[list[float], bytes]] = []
    try:
        for offset in range(0, len(timestamps), frames_per_sheet):
            chunk = timestamps[offset : offset + frames_per_sheet]
            rows = (len(chunk) + columns - 1) // columns
            canvas = np.full(
                (rows * tile_height, columns * tile_width, 3), 235, dtype=np.uint8
            )
            for index, timestamp in enumerate(chunk):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                scale = min(tile_width / width, tile_height / height)
                resized = cv2.resize(
                    frame,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                row, column = divmod(index, columns)
                x = column * tile_width + (tile_width - resized.shape[1]) // 2
                y = row * tile_height + (tile_height - resized.shape[0]) // 2
                canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                label = f"{prefix} {timestamp:.3f}s"
                font_scale = max(0.38, tile_width / 520.0)
                thickness = max(1, round(tile_width / 240))
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                cv2.rectangle(
                    canvas,
                    (column * tile_width, row * tile_height),
                    (
                        min((column + 1) * tile_width, column * tile_width + label_width + 8),
                        min((row + 1) * tile_height, row * tile_height + label_height + 9),
                    ),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (column * tile_width + 4, row * tile_height + label_height + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )
            ok, encoded = cv2.imencode(
                ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 86]
            )
            if not ok:
                raise RuntimeError("Cannot encode a localization contact sheet.")
            sheets.append((chunk, encoded.tobytes()))
    finally:
        capture.release()
    return sheets


def _localization_evidence_timestamps(
    *,
    start: float,
    end: float,
    best: float,
    duration: float,
    count: int,
) -> list[float]:
    """Create a before/during/after evidence sequence around a verified action."""
    if count <= 0:
        return []
    start = min(duration, max(0.0, start))
    end = min(duration, max(start, end))
    best = min(duration, max(start, min(end, best)))
    if end - start < 0.25:
        start = max(0.0, best - 0.75)
        end = min(duration, best + 0.75)
    before_count = count // 3
    during_count = count - 2 * before_count
    after_count = before_count

    def spaced(left: float, right: float, amount: int) -> list[float]:
        if amount <= 0:
            return []
        if amount == 1:
            return [(left + right) / 2.0]
        return [left + (right - left) * i / (amount - 1) for i in range(amount)]

    values = [
        *spaced(max(0.0, start - 2.0), max(0.0, start - 0.125), before_count),
        *spaced(start, end, during_count),
        *spaced(min(duration, end + 0.125), min(duration, end + 2.0), after_count),
    ]
    unique: list[float] = []
    for value in values:
        value = round(min(duration, max(0.0, value)), 3)
        if all(abs(value - existing) > 0.05 for existing in unique):
            unique.append(value)
    return unique[:count]


def _coarse_localization_candidates(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    step: dict[str, Any],
    overview_sheets: list[tuple[list[float], bytes]],
    initial_target: dict[str, Any],
    duration: float,
    candidate_count: int,
    search_radius: float,
    max_tokens: int,
    retries: int,
) -> list[dict[str, Any]]:
    hints = initial_target.get("timestamps", [])
    prompt = f"""
You are the COARSE TEMPORAL LOCALIZER for a clinical procedure video. Find where
the action in this single rubric criterion is attempted. Do not grade correctness.
The contact sheets cover the full {duration:.3f}-second timeline and every thumbnail
has its exact timestamp.

Criterion:
{json.dumps(step, ensure_ascii=False)}

Weak timestamps from earlier models (search hints only; they may be wrong):
{json.dumps(hints)}

Return up to {candidate_count} distinct candidate intervals ranked from best to
worst. Each interval should normally extend about {search_radius:.1f} seconds on
both sides of the most relevant thumbnail. Include incomplete or incorrect attempts.
If the action cannot be localized, return an empty candidates list. Do not treat a
missing action as an incorrect performance.

Return JSON only:
{{
  "candidates": [
    {{
      "start_seconds": 0.0,
      "end_seconds": 10.0,
      "anchor_seconds": 5.0,
      "confidence": 0.0,
      "reason": "visible action or context used for localization"
    }}
  ]
}}
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, (timestamps, jpeg) in enumerate(overview_sheets, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Full-timeline contact sheet {index}/{len(overview_sheets)}; "
                    f"timestamps {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s"
                ),
            }
        )
        content.append(_image_item(jpeg))
    raw = _request_json(
        client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Locate an attempted action in a complete video timeline. "
                    "Never grade it and never convert absence into failure. Return JSON only."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        retries=retries,
        image_detail="high",
    )
    candidates: list[dict[str, Any]] = []
    for item in raw.get("candidates", []):
        if not isinstance(item, dict):
            continue
        try:
            anchor = float(item.get("anchor_seconds"))
            start = float(item.get("start_seconds", anchor - search_radius))
            end = float(item.get("end_seconds", anchor + search_radius))
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            continue
        anchor = min(duration, max(0.0, anchor))
        start = min(anchor, max(0.0, start))
        end = max(anchor, min(duration, end))
        if end - start < 1.0:
            start = max(0.0, anchor - search_radius)
            end = min(duration, anchor + search_radius)
        if any(abs(anchor - old["anchor_seconds"]) < 2.0 for old in candidates):
            continue
        candidates.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "anchor_seconds": round(anchor, 3),
                "confidence": round(confidence, 4),
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        if len(candidates) >= candidate_count:
            break
    return candidates


def _fine_localize_criterion(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    raw_video: Path,
    step: dict[str, Any],
    candidates: list[dict[str, Any]],
    refinement_fps: float,
    confidence_threshold: float,
    duration: float,
    max_image_side: int,
    evidence_images: int,
    max_tokens: int,
    retries: int,
) -> dict[str, Any]:
    fine_sets: list[tuple[int, list[float], bytes]] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        start = float(candidate["start_seconds"])
        end = float(candidate["end_seconds"])
        count = max(2, int(round((end - start) * refinement_fps)) + 1)
        timestamps = [
            round(start + (end - start) * index / (count - 1), 3)
            for index in range(count)
        ]
        for chunk, jpeg in _contact_sheets(
            raw_video,
            timestamps,
            max_side=max_image_side,
            prefix=f"C{candidate_index}",
        ):
            fine_sets.append((candidate_index, chunk, jpeg))

    prompt = f"""
You are the FINE TEMPORAL LOCALIZER for one clinical procedure criterion. The
candidate contact sheets densely sample possible intervals at {refinement_fps:g} FPS.
Find the interval that actually shows an attempt at the criterion. Localize only;
do not decide the rubric score.

Criterion:
{json.dumps(step, ensure_ascii=False)}

Candidate intervals:
{json.dumps(candidates, ensure_ascii=False)}

Rules:
- An attempt may be correct, partial, or incorrect; all count as action_visible=true.
- Use only timestamps printed in the supplied contact sheets for evidence_seconds.
- Weak surrounding context alone is insufficient.
- If none of the candidates directly shows the action, return action_visible=false.
- Never infer failure from an absent action.

Return JSON only:
{{
  "action_visible": true,
  "candidate_rank": 1,
  "start_seconds": 0.0,
  "end_seconds": 1.0,
  "best_timestamp": 0.5,
  "evidence_seconds": [0.5],
  "confidence": 0.0,
  "reason": "direct visual localization evidence"
}}
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, (candidate_index, timestamps, jpeg) in enumerate(fine_sets, start=1):
        content.append(
            {
                "type": "text",
                "text": (
                    f"Fine sheet {index}/{len(fine_sets)} for candidate "
                    f"C{candidate_index}: {timestamps[0]:.3f}s to {timestamps[-1]:.3f}s"
                ),
            }
        )
        content.append(_image_item(jpeg))
    raw = _request_json(
        client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Confirm the temporal location of an attempted action from dense "
                    "timestamped frames. Do not grade. Return JSON only."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        retries=retries,
        image_detail="high",
    )
    try:
        confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    supplied = sorted({timestamp for _, values, _ in fine_sets for timestamp in values})
    verified: list[float] = []
    tolerance = max(0.08, 0.55 / refinement_fps)
    for value in raw.get("evidence_seconds", []):
        try:
            requested = float(value)
        except (TypeError, ValueError):
            continue
        nearest = min(supplied, key=lambda timestamp: abs(timestamp - requested))
        if abs(nearest - requested) <= tolerance:
            verified.append(nearest)
    action_visible = raw.get("action_visible") is True
    if not action_visible or confidence < confidence_threshold or not verified:
        return {
            "status": "insufficient",
            "action_visible": action_visible,
            "confidence": round(confidence, 4),
            "evidence_seconds": sorted(set(verified)),
            "reason": str(raw.get("reason", "")).strip(),
        }
    try:
        start = float(raw.get("start_seconds", min(verified)))
        end = float(raw.get("end_seconds", max(verified)))
        best = float(raw.get("best_timestamp", verified[len(verified) // 2]))
    except (TypeError, ValueError):
        start, end = min(verified), max(verified)
        best = verified[len(verified) // 2]
    start = min(duration, max(0.0, start))
    end = min(duration, max(start, end))
    best = min(end, max(start, best))
    return {
        "status": "verified",
        "action_visible": True,
        "confidence": round(confidence, 4),
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "best_timestamp": round(best, 3),
        "evidence_seconds": sorted(set(verified)),
        "assessment_timestamps": _localization_evidence_timestamps(
            start=start,
            end=end,
            best=best,
            duration=duration,
            count=evidence_images,
        ),
        "reason": str(raw.get("reason", "")).strip(),
    }


def _localize_assessment_manifest(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    raw_video: Path,
    rubric: dict[str, Any],
    manifest: dict[str, Any],
    cache_dir: Path,
    resume: bool,
    workers: int,
    coarse_seconds: float,
    candidate_count: int,
    search_radius: float,
    refinement_fps: float,
    evidence_images: int,
    confidence_threshold: float,
    max_image_side: int,
    max_tokens: int,
    retries: int,
) -> None:
    """Replace weak timestamp hints with criterion-verified evidence sequences."""
    duration = max(
        (float(item["end_seconds"]) for item in rubric.get("_events", [])),
        default=0.0,
    )
    if duration <= 0:
        duration = float(manifest.get("duration_seconds", 0.0) or 0.0)
    if duration <= 0:
        raise ValueError("Criterion localization requires a positive video duration.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    overview = _contact_sheets(
        raw_video,
        _timeline_timestamps(duration, coarse_seconds),
        max_side=max_image_side,
        prefix="T",
    )
    video_stat = raw_video.stat()
    steps = {int(step["id"]): step for step in rubric["steps"]}

    def localize(target: dict[str, Any]) -> dict[str, Any]:
        step = steps[int(target["step_id"])]
        cache_material = json.dumps(
            {
                "video": str(raw_video),
                "video_size": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "rubric_id": rubric["rubric_id"],
                "step": step,
                "initial_target": target,
                "provider": provider,
                "model": model,
                "coarse_seconds": coarse_seconds,
                "candidate_count": candidate_count,
                "search_radius": search_radius,
                "refinement_fps": refinement_fps,
                "evidence_images": evidence_images,
                "confidence_threshold": confidence_threshold,
                "prompt_version": "criterion_localizer_v1",
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()[:20]
        path = cache_dir / f"step_{int(target['step_id']):02d}_{key}.json"
        if resume and path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            result["cache_hit"] = True
        else:
            candidates = _coarse_localization_candidates(
                client,
                provider=provider,
                model=model,
                step=step,
                overview_sheets=overview,
                initial_target=target,
                duration=duration,
                candidate_count=candidate_count,
                search_radius=search_radius,
                max_tokens=max_tokens,
                retries=retries,
            )
            if candidates:
                result = _fine_localize_criterion(
                    client,
                    provider=provider,
                    model=model,
                    raw_video=raw_video,
                    step=step,
                    candidates=candidates,
                    refinement_fps=refinement_fps,
                    confidence_threshold=confidence_threshold,
                    duration=duration,
                    max_image_side=max_image_side,
                    evidence_images=evidence_images,
                    max_tokens=max_tokens,
                    retries=retries,
                )
            else:
                result = {
                    "status": "insufficient",
                    "action_visible": False,
                    "confidence": 0.0,
                    "evidence_seconds": [],
                    "reason": "The coarse full-timeline search found no candidate interval.",
                }
            result["candidates"] = candidates
            result["cache_hit"] = False
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        updated = dict(target)
        updated["initial_selection_basis"] = target.get("selection_basis")
        updated["initial_timestamps"] = target.get("timestamps", [])
        updated["selection_basis"] = "criterion_localizer"
        updated["localization"] = result
        updated["timestamps"] = (
            result.get("assessment_timestamps", [])
            if result.get("status") == "verified"
            else []
        )
        print(
            f"Localization: criterion {target['step_id']} "
            f"{result.get('status')} (confidence={result.get('confidence', 0):.2f}, "
            f"{'cache' if result.get('cache_hit') else 'API'})"
        )
        return updated

    localized: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(localize, target) for target in manifest["targets"]]
        for future in as_completed(futures):
            localized.append(future.result())
    manifest["targets"] = sorted(localized, key=lambda item: item["step_id"])
    manifest["selection_policy"]["criterion_localization"] = {
        "enabled": True,
        "coarse_interval_seconds": coarse_seconds,
        "candidate_windows": candidate_count,
        "search_radius_seconds": search_radius,
        "refinement_fps": refinement_fps,
        "evidence_images": evidence_images,
        "confidence_threshold": confidence_threshold,
        "workers": workers,
    }
    manifest["summary"].update(
        {
            "localization_verified_steps": sum(
                item.get("localization", {}).get("status") == "verified"
                for item in localized
            ),
            "localization_insufficient_steps": sum(
                item.get("localization", {}).get("status") != "verified"
                for item in localized
            ),
            "selected_images": sum(len(item["timestamps"]) for item in localized),
        }
    )


def _apply_localization_sequence_guard(
    manifest: dict[str, Any], *, tolerance_seconds: float = 5.0
) -> None:
    """Reject localized outliers using the longest rubric-ordered timeline path."""
    localized = [
        item
        for item in manifest.get("targets", [])
        if item.get("localization", {}).get("status") == "verified"
        and isinstance(item.get("localization", {}).get("best_timestamp"), (int, float))
    ]
    localized.sort(key=lambda item: int(item["step_id"]))
    if len(localized) < 2:
        return

    # Each state maximizes the number of chronologically consistent rubric steps;
    # summed confidence breaks equal-length ties without rewarding one outlier.
    states: list[tuple[int, float, int | None]] = []
    for index, item in enumerate(localized):
        best = (1, float(item["localization"].get("confidence", 0.0)), None)
        current_time = float(item["localization"]["best_timestamp"])
        for previous_index in range(index):
            previous_time = float(
                localized[previous_index]["localization"]["best_timestamp"]
            )
            if current_time + tolerance_seconds < previous_time:
                continue
            previous_state = states[previous_index]
            candidate = (
                previous_state[0] + 1,
                previous_state[1]
                + float(item["localization"].get("confidence", 0.0)),
                previous_index,
            )
            if candidate[:2] > best[:2]:
                best = candidate
        states.append(best)

    cursor = max(range(len(states)), key=lambda index: states[index][:2])
    accepted_indexes: set[int] = set()
    while cursor is not None:
        accepted_indexes.add(cursor)
        cursor = states[cursor][2]
    rejected_step_ids: list[int] = []
    for index, target in enumerate(localized):
        if index in accepted_indexes:
            target["localization"]["sequence_consistent"] = True
            continue
        localization = target["localization"]
        localization["sequence_consistent"] = False
        localization["status"] = "sequence_inconsistent"
        localization["reason"] = (
            f"{str(localization.get('reason', '')).strip()} "
            "Rejected because this timestamp is an outlier from the longest "
            "rubric-ordered timeline path."
        ).strip()
        target["timestamps"] = []
        rejected_step_ids.append(int(target["step_id"]))

    manifest["selection_policy"]["criterion_localization"].update(
        {
            "sequence_guard": "longest_rubric_ordered_path",
            "sequence_tolerance_seconds": tolerance_seconds,
        }
    )
    manifest["summary"].update(
        {
            "localization_sequence_rejected_steps": len(rejected_step_ids),
            "localization_sequence_rejected_step_ids": rejected_step_ids,
            "localization_verified_steps": sum(
                item.get("localization", {}).get("status") == "verified"
                for item in manifest["targets"]
            ),
            "localization_insufficient_steps": sum(
                item.get("localization", {}).get("status") != "verified"
                for item in manifest["targets"]
            ),
            "selected_images": sum(
                len(item.get("timestamps", [])) for item in manifest["targets"]
            ),
        }
    )


def _build_assessment_image_manifest(
    *,
    mode: str,
    rubric: dict[str, Any],
    event_results: list[dict[str, Any]],
    preliminary_assessment: dict[str, Any],
    uncertainty_threshold: float,
    images_per_criterion: int,
    context_frames: int,
    critical_criterion_images: int,
    rescue_step_limit: int,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable visual verification plan."""
    if mode not in ASSESSMENT_IMAGE_MODES:
        raise ValueError(f"Unknown assessment image mode: {mode}")
    duration = max(
        (float(event["end_seconds"]) for event in event_results), default=0.0
    )
    unresolved = {
        int(item["step_id"])
        for item in preliminary_assessment["steps"]
        if item.get("score") is None
    }
    preliminary_by_id = {
        int(item["step_id"]): item
        for item in preliminary_assessment["steps"]
        if isinstance(item, dict) and isinstance(item.get("step_id"), int)
    }
    critical_ids = _critical_step_ids(rubric)
    targets: list[dict[str, Any]] = []
    for step in rubric["steps"]:
        step_id = int(step["id"])
        direct = _direct_timestamps_for_step(step_id, event_results)
        if mode == "never":
            continue
        if mode == "rescue" and step_id not in unresolved:
            continue
        preliminary = preliminary_by_id.get(step_id, {})
        confidence = float(preliminary.get("confidence", 0.0) or 0.0)
        uncertain = (
            preliminary.get("score") is None
            or bool(preliminary.get("requires_review"))
            or confidence < uncertainty_threshold
        )
        if mode == "uncertain" and not uncertain:
            continue
        if mode == "evidence" and not direct:
            continue

        selection_basis = "observation_timestamp"
        anchors = direct
        if not anchors:
            selection_basis = "temporal_prior"
            anchors = [
                _candidate_timestamp_for_step(
                    step_id,
                    preliminary_assessment,
                    event_results,
                    len(rubric["steps"]),
                )
            ]
        limit = (
            critical_criterion_images
            if step_id in critical_ids
            else images_per_criterion
        )
        timestamps = _select_assessment_timestamps(
            anchors,
            duration=duration,
            max_images=limit,
            context_frames=context_frames,
        )
        if timestamps:
            targets.append(
                {
                    "step_id": step_id,
                    "critical": step_id in critical_ids,
                    "selection_basis": selection_basis,
                    "selection_reason": (
                        "text_first_uncertain" if mode == "uncertain" else mode
                    ),
                    "text_first_confidence": round(confidence, 4),
                    "timestamps": timestamps,
                }
            )
    if mode == "rescue":
        targets = targets[:rescue_step_limit]
    return {
        "schema_version": "1.0",
        "mode": mode,
        "duration_seconds": duration,
        "selection_policy": {
            "images_per_criterion": images_per_criterion,
            "context_frames": context_frames,
            "context_interval_seconds": 1.0,
            "critical_criterion_images": critical_criterion_images,
            "rescue_step_limit": rescue_step_limit,
            "uncertainty_threshold": uncertainty_threshold,
            "uses_raw_frames": True,
            "observation_text_sent_to_visual_review": False,
        },
        "targets": targets,
        "summary": {
            "selected_steps": len(targets),
            "selected_images": sum(len(item["timestamps"]) for item in targets),
        },
    }


def _audio_for_visual_step(
    step: dict[str, Any],
    timestamps: list[float],
    audio_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    if "audio" not in step.get("modalities", []) or not audio_evidence:
        return {"enabled": False}
    chunks = []
    for chunk in audio_evidence.get("chunks", []):
        start = float(chunk.get("start_seconds", 0.0) or 0.0)
        end = float(chunk.get("end_seconds", start) or start)
        if any(start - 15.0 <= timestamp <= end + 15.0 for timestamp in timestamps):
            chunks.append(chunk)
    return {"enabled": bool(chunks), "chunks": chunks}


def _review_step_with_images(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    raw_video: Path,
    rubric: dict[str, Any],
    target: dict[str, Any],
    max_image_side: int,
    max_tokens: int,
    retries: int,
    image_detail: str,
    audio_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Independently verify one criterion from raw frames, not observation prose."""
    step_id = int(target["step_id"])
    step = next(item for item in rubric["steps"] if int(item["id"]) == step_id)
    relevant_rules = [
        rule
        for rule in rubric.get("critical_rules") or []
        if not rule.get("relevant_step_ids")
        or step_id in rule.get("relevant_step_ids", [])
    ]
    prompt = f"""
You are an INDEPENDENT VISUAL RUBRIC VERIFIER. Judge one criterion from the raw
video frames below. Observation-model descriptions are deliberately excluded;
do not assume that another model's interpretation is correct.

Criterion:
{json.dumps(step, ensure_ascii=False)}

Relevant global scoring context:
{json.dumps({
    "assessment_policy": rubric.get("assessment_policy", {}),
    "sequence_constraints": rubric.get("sequence_constraints", []),
    "score_scale": rubric.get("score_scale", {}),
    "score_calculation": rubric.get("score_calculation", {}),
}, ensure_ascii=False)}

Relevant critical rules:
{json.dumps(relevant_rules, ensure_ascii=False)}

Nearby audio evidence, if the criterion permits audio:
{json.dumps(_audio_for_visual_step(step, target["timestamps"], audio_evidence), ensure_ascii=False)}

Rules:
- Treat each image as evidence only for its labeled timestamp.
- Use only a score in allowed_scores, or score=null when the supplied frames do
  not prove the criterion.
- Missing evidence is not proof of failure. Use score 0 only for a directly
  supported error or omission under the criterion's scoring rules.
- criterion_action_visible=true only when the supplied sequence directly shows
  an attempt at this criterion, rather than a nearby or similar action.
- direct_error_visible=true only when the sequence positively shows an incorrect
  action or a skipped action across a visible before-to-after transition.
- Apply scoring_rules and do_not_infer exactly. Do not invent deductions.
- A temporal-prior frame is only a search hint, not evidence that the action occurred.
- List only supplied timestamps in evidence_seconds. Use an empty list when no
  supplied image supports the judgment.

Return JSON only:
{{
  "steps": [{{
    "step_id": {step_id},
    "criterion_action_visible": true,
    "direct_error_visible": false,
    "score": 2,
    "verdict": "complete|partial|incorrect|unverifiable",
    "confidence": 0.0,
    "evidence_ids": [],
    "audio_evidence_ids": [],
    "evidence_seconds": [0.0],
    "score_basis": "direct_visual|contextual_inference",
    "reason": "brief image-grounded rationale"
  }}]
}}
""".strip()
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    capture = cv2.VideoCapture(str(raw_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open the raw video: {raw_video}")
    try:
        for timestamp in target["timestamps"]:
            content.append(
                {
                    "type": "text",
                    "text": f"Raw frame for criterion {step_id} at {timestamp:.3f}s",
                }
            )
            content.append(_image_item(_read_frame(capture, timestamp, max_image_side)))
    finally:
        capture.release()

    result = _request_json(
        client,
        provider=provider,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Independently verify rubric criteria from raw images. "
                    "Insufficient evidence must remain unverifiable. Return JSON only."
                ),
            },
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        retries=retries,
        image_detail=image_detail,
    )
    item = next(
        (
            value
            for value in result.get("steps", [])
            if isinstance(value, dict) and value.get("step_id") == step_id
        ),
        None,
    )
    if item is None:
        raise ValueError(f"Visual review returned no result for criterion {step_id}")
    item = _enforce_visual_evidence_contract(item)
    supplied = target["timestamps"]
    verified_seconds: list[float] = []
    for value in item.get("evidence_seconds", []):
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        match = next(
            (timestamp for timestamp in supplied if abs(timestamp - candidate) <= 0.25),
            None,
        )
        if match is not None:
            verified_seconds.append(match)
    item["evidence_seconds"] = sorted(set(verified_seconds))
    item["assessment_image_selection_basis"] = target["selection_basis"]
    return {"steps": [item]}


def _enforce_visual_evidence_contract(item: dict[str, Any]) -> dict[str, Any]:
    """Make the no-evidence-is-not-failure rule deterministic after model output."""
    result = dict(item)
    action_visible = result.get("criterion_action_visible") is True
    direct_error_visible = result.get("direct_error_visible") is True
    invalid_zero = result.get("score") == 0 and not direct_error_visible
    if not action_visible or invalid_zero:
        original_reason = str(result.get("reason", "")).strip()
        if not action_visible:
            contract_reason = (
                "The supplied sequence does not directly show the criterion action; "
                "absence cannot be scored as failure."
            )
        else:
            contract_reason = (
                "No directly visible error supports score 0; missing evidence cannot "
                "be scored as failure."
            )
        result.update(
            {
                "score": None,
                "verdict": "unverifiable",
                "confidence": 0.0,
                "evidence_seconds": [],
                "score_basis": "contextual_inference",
                "reason": f"{original_reason} {contract_reason}".strip(),
                "evidence_contract_adjusted": True,
            }
        )
    else:
        result["evidence_contract_adjusted"] = False
    return result


def _run_assessment_image_reviews(
    client: OpenAI,
    *,
    provider: str,
    model: str,
    base_url: str | None,
    raw_video: Path,
    rubric: dict[str, Any],
    manifest: dict[str, Any],
    cache_dir: Path,
    resume: bool,
    workers: int,
    max_image_side: int,
    max_tokens: int,
    retries: int,
    image_detail: str,
    audio_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run bounded per-criterion visual reviews and preserve recoverable failures."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_stat = raw_video.stat()

    def review(target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        cache_material = json.dumps(
            {
                "video": str(raw_video),
                "video_size": video_stat.st_size,
                "video_mtime_ns": video_stat.st_mtime_ns,
                "rubric_id": rubric["rubric_id"],
                "assessment_rubric": _assessment_rubric(rubric),
                "step": next(
                    step for step in rubric["steps"] if step["id"] == target["step_id"]
                ),
                # Keep cache identity tied to the actual visual request. Selection
                # reasons and text confidence are audit metadata and must not
                # invalidate an otherwise identical image review.
                "target": {
                    key: target[key]
                    for key in (
                        "step_id",
                        "critical",
                        "selection_basis",
                        "timestamps",
                    )
                },
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "prompt_version": "independent_visual_v2_action_contract",
                "max_image_side": max_image_side,
                "max_tokens": max_tokens,
                "image_detail": image_detail,
                "audio": _audio_for_visual_step(
                    next(
                        step
                        for step in rubric["steps"]
                        if step["id"] == target["step_id"]
                    ),
                    target["timestamps"],
                    audio_evidence,
                ),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()[:20]
        cache_path = cache_dir / f"step_{target['step_id']:02d}_{key}.json"
        record = dict(target)
        try:
            if resume and cache_path.is_file():
                result = json.loads(cache_path.read_text(encoding="utf-8"))
                record.update({"status": "completed", "cache_hit": True})
            else:
                result = _review_step_with_images(
                    client,
                    provider=provider,
                    model=model,
                    raw_video=raw_video,
                    rubric=rubric,
                    target=target,
                    max_image_side=max_image_side,
                    max_tokens=max_tokens,
                    retries=retries,
                    image_detail=image_detail,
                    audio_evidence=audio_evidence,
                )
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                record.update({"status": "completed", "cache_hit": False})
            print(
                f"Assessment images: criterion {target['step_id']} "
                f"({'cache' if record['cache_hit'] else 'API'})"
            )
            return record, result
        except Exception as error:  # Keep the valid text-only judgment available.
            record.update(
                {"status": "failed", "cache_hit": False, "error": str(error)}
            )
            print(f"Assessment images: criterion {target['step_id']} failed: {error}")
            return record, None

    skipped = [target for target in manifest["targets"] if not target.get("timestamps")]
    reviewable = [target for target in manifest["targets"] if target.get("timestamps")]
    records: list[dict[str, Any]] = [
        {
            **target,
            "status": "skipped_localization",
            "cache_hit": bool(target.get("localization", {}).get("cache_hit")),
        }
        for target in skipped
    ]
    replacements: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(review, target) for target in reviewable]
        for future in as_completed(futures):
            record, result = future.result()
            records.append(record)
            if result:
                replacements.extend(result.get("steps", []))
    manifest["targets"] = sorted(records, key=lambda item: item["step_id"])
    manifest["summary"].update(
        {
            "completed_steps": sum(item["status"] == "completed" for item in records),
            "failed_steps": sum(item["status"] == "failed" for item in records),
            "localization_skipped_steps": sum(
                item["status"] == "skipped_localization" for item in records
            ),
            "cache_hits": sum(bool(item["cache_hit"]) for item in records),
        }
    )
    return {"steps": sorted(replacements, key=lambda item: item["step_id"])}


def _valid_score(item: dict[str, Any]) -> int | None:
    """Return a score only when the model's score and verdict agree."""
    score = item.get("score")
    verdict = str(item.get("verdict", "unverifiable"))
    if isinstance(score, bool) or score not in {0, 1, 2}:
        return None
    if SCORE_BY_VERDICT.get(verdict) != score:
        return None
    return int(score)


def _confidence(item: dict[str, Any]) -> float:
    try:
        return min(1.0, max(0.0, float(item.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _fuse_step_judgments(
    text_first: dict[str, Any],
    visual_review: dict[str, Any],
    preliminary_assessment: dict[str, Any],
    rubric: dict[str, Any],
    *,
    visual_override_confidence: float,
    visual_override_margin: float,
) -> dict[str, Any]:
    """Fuse text and raw-image judgments while retaining a per-step audit trail."""
    visual_by_id = {
        item.get("step_id"): item
        for item in visual_review.get("steps", [])
        if isinstance(item, dict) and isinstance(item.get("step_id"), int)
    }
    preliminary_by_id = {
        item.get("step_id"): item
        for item in preliminary_assessment.get("steps", [])
        if isinstance(item, dict) and isinstance(item.get("step_id"), int)
    }
    allowed_by_id = {
        int(step["id"]): set(step.get("allowed_scores", [0, 1, 2]))
        for step in rubric["steps"]
    }
    merged = dict(text_first)
    merged_steps: list[dict[str, Any]] = []
    seen_step_ids: set[int] = set()
    for original in text_first.get("steps", []):
        if not isinstance(original, dict):
            continue
        step_id = original.get("step_id")
        seen_step_ids.add(step_id)
        visual = visual_by_id.get(step_id)
        if visual is None:
            merged_steps.append(original)
            continue

        preliminary = preliminary_by_id.get(step_id, {})
        text_score = preliminary.get("score")
        text_confidence = _confidence(preliminary)
        visual_score = _valid_score(visual)
        visual_confidence = _confidence(visual)
        visual_has_evidence = bool(
            visual.get("evidence_seconds") or visual.get("audio_evidence_ids")
        )
        if visual_score not in allowed_by_id.get(step_id, {0, 1, 2}):
            visual_score = None

        selected = dict(original)
        decision = "text_retained_visual_insufficient"
        if text_score is None and visual_score is not None and visual_has_evidence:
            selected = dict(visual)
            decision = "visual_rescue"
        elif visual_score is None or not visual_has_evidence:
            decision = "text_retained_visual_insufficient"
        elif visual_score == text_score:
            selected["confidence"] = max(_confidence(original), visual_confidence)
            selected["evidence_seconds"] = sorted(
                {
                    *selected.get("evidence_seconds", []),
                    *visual.get("evidence_seconds", []),
                }
            )
            selected["score_basis"] = "fused_evidence"
            decision = "text_confirmed_by_visual"
        elif (
            visual_confidence >= visual_override_confidence
            and visual_confidence >= text_confidence + visual_override_margin
        ):
            selected = dict(visual)
            decision = "visual_override_strong_evidence"
        else:
            decision = "text_retained_visual_disagreement"

        selected.update(
            {
                "fusion_decision": decision,
                "text_first_score": text_score,
                "text_first_confidence": round(text_confidence, 4),
                "visual_review_score": visual_score,
                "visual_review_confidence": round(visual_confidence, 4),
                "visual_review_reason": str(visual.get("reason", "")).strip(),
            }
        )
        merged_steps.append(selected)
    for step_id, visual in visual_by_id.items():
        if step_id in seen_step_ids:
            continue
        visual_score = _valid_score(visual)
        if visual_score not in allowed_by_id.get(step_id, {0, 1, 2}):
            visual_score = None
        selected = dict(visual)
        selected.update(
            {
                "fusion_decision": (
                    "visual_rescue"
                    if visual_score is not None
                    and bool(
                        visual.get("evidence_seconds")
                        or visual.get("audio_evidence_ids")
                    )
                    else "text_missing_visual_insufficient"
                ),
                "text_first_score": None,
                "text_first_confidence": 0.0,
                "visual_review_score": visual_score,
                "visual_review_confidence": round(_confidence(visual), 4),
                "visual_review_reason": str(visual.get("reason", "")).strip(),
            }
        )
        merged_steps.append(selected)
    merged["steps"] = merged_steps
    return merged


def _normalize_assessment(
    raw: dict[str, Any],
    rubric: dict[str, Any],
    event_results: list[dict[str, Any]],
    confidence_threshold: float,
    assessment_policy: str = "conservative",
    audio_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if assessment_policy not in {"conservative", "lenient"}:
        raise ValueError("assessment_policy must be conservative or lenient.")
    lenient = assessment_policy == "lenient"
    evidence = {
        observation["observation_id"]: observation
        for event in event_results
        for observation in event["observations"]
    }
    raw_by_id = {
        item.get("step_id"): item
        for item in raw.get("steps", [])
        if isinstance(item, dict) and isinstance(item.get("step_id"), int)
    }
    audio_by_id = {
        str(item["audio_id"]): item
        for item in (audio_evidence or {}).get("chunks", [])
        if isinstance(item, dict) and item.get("audio_id") and item.get("usable")
    }
    results: list[dict[str, Any]] = []
    for step in rubric["steps"]:
        item = raw_by_id.get(step["id"], {})
        verdict = str(item.get("verdict", "unverifiable"))
        if verdict not in SCORE_BY_VERDICT:
            verdict = "unverifiable"
        expected_score = SCORE_BY_VERDICT[verdict]
        allowed_scores = set(step.get("allowed_scores", [0, 1, 2]))
        model_score = item.get("score")
        if isinstance(model_score, bool) or model_score not in {0, 1, 2, None}:
            model_score = None
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence_ids = [
            value
            for value in item.get("evidence_ids", [])
            if isinstance(value, str) and value in evidence
        ]
        audio_evidence_ids = [
            value
            for value in item.get("audio_evidence_ids", [])
            if isinstance(value, str)
            and value in audio_by_id
            and "audio" in step.get("modalities", [])
        ]
        visual_evidence_seconds: list[float] = []
        for value in item.get("evidence_seconds", []):
            try:
                visual_evidence_seconds.append(round(float(value), 3))
            except (TypeError, ValueError):
                continue
        visual_evidence_seconds = sorted(set(visual_evidence_seconds))
        review_reasons: list[str] = []
        if lenient:
            if model_score in {0, 1, 2}:
                score = int(model_score)
                verdict = {0: "incorrect", 1: "partial", 2: "complete"}[score]
                if expected_score != score:
                    review_reasons.append(
                        "The model score and verdict were normalized to the score."
                    )
            elif expected_score in {0, 1, 2}:
                score = int(expected_score)
            else:
                score = 1
                verdict = "partial"
                review_reasons.append(
                    "No decisive model score; lenient policy assigned partial credit."
                )
        else:
            score = expected_score
            if model_score != expected_score:
                review_reasons.append("The VLM score and verdict are inconsistent.")
                score = None
                verdict = "unverifiable"

        if score is not None and score not in allowed_scores:
            review_reasons.append(
                "The model score is not legal for this rubric criterion."
            )
            score = None
            verdict = "unverifiable"

        selected_evidence = [evidence[value] for value in evidence_ids]
        if visual_evidence_seconds:
            visual_id = f"assessment_visual_step_{step['id']:02d}"
            evidence_ids.append(visual_id)
            selected_evidence.append(
                {
                    "observation_id": visual_id,
                    "event_id": "assessment_visual_rescue",
                    "step_id": step["id"],
                    "status": verdict,
                    "confidence": round(confidence, 4),
                    "evidence_seconds": visual_evidence_seconds,
                    "description": str(item.get("reason", "")).strip(),
                }
            )

        selected_audio = [audio_by_id[value] for value in audio_evidence_ids]
        if expected_score is not None and not selected_evidence and not selected_audio:
            review_reasons.append("No valid direct visual or audio evidence was provided.")
            if not lenient:
                score = None
                verdict = "unverifiable"
        if lenient and not selected_evidence and not selected_audio:
            review_reasons.append("The score is inferred without direct evidence.")
        if confidence < confidence_threshold:
            review_reasons.append("Confidence is below the acceptance threshold.")
            if not lenient:
                score = None
                verdict = "unverifiable"
        score_basis = str(item.get("score_basis", "")).strip()
        if score_basis not in {
            "observation_evidence",
            "direct_visual",
            "fused_evidence",
            "contextual_inference",
        }:
            if visual_evidence_seconds:
                score_basis = "direct_visual"
            elif selected_evidence:
                score_basis = "observation_evidence"
            else:
                score_basis = "contextual_inference"
        requires_review = score is None or (
            lenient
            and (
                bool(review_reasons)
                or score_basis == "contextual_inference"
            )
        )
        results.append(
            {
                "step_id": step["id"],
                "key": step.get("key"),
                "description": step.get("description"),
                "score": score,
                "max_score": int(step.get("max_score", 2)),
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "score_basis": score_basis,
                "evidence_ids": evidence_ids,
                "evidence": selected_evidence,
                "audio_evidence_ids": audio_evidence_ids,
                "audio_evidence": selected_audio,
                "reason": str(item.get("reason", "")).strip(),
                "requires_review": requires_review,
                "review_reasons": review_reasons,
                "fusion_decision": item.get("fusion_decision"),
                "text_first_score": item.get("text_first_score"),
                "text_first_confidence": item.get("text_first_confidence"),
                "visual_review_score": item.get("visual_review_score"),
                "visual_review_confidence": item.get("visual_review_confidence"),
                "visual_review_reason": item.get("visual_review_reason"),
            }
        )

    sequence_violations: list[dict[str, Any]] = []
    previous: tuple[int, float] | None = None
    for result in results:
        timestamps = [
            timestamp
            for observation in result["evidence"]
            for timestamp in observation["evidence_seconds"]
        ]
        if result["score"] is None or not timestamps:
            continue
        first_time = min(timestamps)
        if previous is not None and first_time + 0.5 < previous[1]:
            sequence_violations.append(
                {
                    "previous_step_id": previous[0],
                    "current_step_id": result["step_id"],
                    "previous_time": previous[1],
                    "current_time": first_time,
                }
            )
        if previous is None or first_time >= previous[1]:
            previous = (result["step_id"], first_time)

    scored = [result for result in results if result["score"] is not None]
    confirmed = sum(result["score"] for result in scored)
    unresolved = len(results) - len(scored)
    inferred_steps = sum(
        result["score_basis"] == "contextual_inference" for result in results
    )
    review_steps = sum(bool(result["requires_review"]) for result in results)
    maximum = sum(int(result["max_score"]) for result in results)
    unresolved_maximum = sum(
        int(result["max_score"]) for result in results if result["score"] is None
    )
    summary = {
        "official_score": confirmed if unresolved == 0 else None,
        "maximum_score": maximum,
        "confirmed_score": confirmed,
        "score_range": [confirmed, confirmed + unresolved_maximum],
        "scored_steps": len(scored),
        "unverifiable_steps": unresolved,
        "assessment_policy": assessment_policy,
        "inferred_steps": inferred_steps,
        "review_steps": review_steps,
        "requires_human_review": (
            unresolved > 0 or review_steps > 0 or bool(sequence_violations)
        ),
    }
    return {
        "summary": summary,
        "steps": results,
        "sequence_violations": sequence_violations,
        "overall_feedback": str(raw.get("overall_feedback", "")).strip(),
        "safety_concerns": [
            str(value) for value in raw.get("safety_concerns", []) if str(value)
        ],
    }


def _write_scores(path: Path, assessment: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "step_id",
            "key",
            "description",
            "score",
            "max_score",
            "verdict",
            "confidence",
            "score_basis",
            "fusion_decision",
            "text_first_score",
            "visual_review_score",
            "requires_review",
            "evidence_seconds",
            "audio_evidence_ids",
            "reason",
            "review_reasons",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in assessment["steps"]:
            row = {key: item.get(key) for key in fieldnames}
            row["evidence_seconds"] = ";".join(
                str(timestamp)
                for observation in item["evidence"]
                for timestamp in observation["evidence_seconds"]
            )
            row["review_reasons"] = "; ".join(item["review_reasons"])
            row["audio_evidence_ids"] = ";".join(item["audio_evidence_ids"])
            writer.writerow(row)


def run_vlm_assessment(
    *,
    raw_video: Path,
    perception: dict[str, Any],
    rubric_path: Path,
    output_dir: Path,
    observation_api_key: str,
    observation_provider: str,
    observation_base_url: str | None,
    observation_model: str,
    assessment_api_key: str,
    assessment_provider: str,
    assessment_base_url: str | None,
    assessment_model: str,
    frames_per_event: int = 8,
    max_image_side: int = 1280,
    include_annotated_frames: bool = False,
    confidence_threshold: float = 0.6,
    assessment_policy: str = "conservative",
    assessment_images: str = "never",
    assessment_image_confidence_threshold: float = 0.75,
    assessment_visual_override_confidence: float = 0.85,
    assessment_visual_override_margin: float = 0.15,
    assessment_images_per_criterion: int = 3,
    assessment_context_frames: int = 1,
    critical_criterion_images: int = 6,
    assessment_image_workers: int = 4,
    assessment_rescue_images: int = 30,
    criterion_localization: bool = False,
    localization_candidates: int = 3,
    localization_coarse_seconds: float = 5.0,
    localization_search_radius: float = 5.0,
    localization_refinement_fps: float = 8.0,
    localization_evidence_images: int = 12,
    localization_confidence_threshold: float = 0.70,
    localization_workers: int = 4,
    event_max_tokens: int = 2000,
    judge_max_tokens: int = 6000,
    retries: int = 2,
    resume: bool = True,
    image_detail: str = "high",
    before_observation: Callable[[], None] | None = None,
    before_assessment: Callable[[], None] | None = None,
    audio_evidence: dict[str, Any] | None = None,
    observation_workers: int = 4,
) -> dict[str, Any]:
    """Extract observations, score the rubric, then independently verify with images."""
    if not observation_api_key:
        raise ValueError("The observation API key must not be empty.")
    if not assessment_api_key:
        raise ValueError("The assessment API key must not be empty.")
    observation_provider = normalize_provider(observation_provider)
    assessment_provider = normalize_provider(assessment_provider)
    if image_detail not in {"low", "high", "original", "auto"}:
        raise ValueError("image_detail must be low, high, original, or auto.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if assessment_policy not in {"conservative", "lenient"}:
        raise ValueError("assessment_policy must be conservative or lenient.")
    if assessment_images not in ASSESSMENT_IMAGE_MODES:
        raise ValueError(
            "assessment_images must be never, rescue, uncertain, evidence, or always."
        )
    if not 0.0 <= assessment_image_confidence_threshold <= 1.0:
        raise ValueError(
            "assessment_image_confidence_threshold must be between 0 and 1."
        )
    if not 0.0 <= assessment_visual_override_confidence <= 1.0:
        raise ValueError(
            "assessment_visual_override_confidence must be between 0 and 1."
        )
    if not 0.0 <= assessment_visual_override_margin <= 1.0:
        raise ValueError("assessment_visual_override_margin must be between 0 and 1.")
    if assessment_images_per_criterion <= 0:
        raise ValueError("assessment_images_per_criterion must be positive.")
    if assessment_context_frames < 0:
        raise ValueError("assessment_context_frames must be non-negative.")
    if critical_criterion_images <= 0:
        raise ValueError("critical_criterion_images must be positive.")
    if not 1 <= assessment_image_workers <= 16:
        raise ValueError("assessment_image_workers must be between 1 and 16.")
    if assessment_rescue_images <= 0:
        raise ValueError("assessment_rescue_images must be positive.")
    if localization_candidates <= 0:
        raise ValueError("localization_candidates must be positive.")
    if localization_coarse_seconds <= 0:
        raise ValueError("localization_coarse_seconds must be positive.")
    if localization_search_radius <= 0:
        raise ValueError("localization_search_radius must be positive.")
    if localization_refinement_fps <= 0:
        raise ValueError("localization_refinement_fps must be positive.")
    if localization_evidence_images <= 0:
        raise ValueError("localization_evidence_images must be positive.")
    if not 0.0 <= localization_confidence_threshold <= 1.0:
        raise ValueError("localization_confidence_threshold must be between 0 and 1.")
    if not 1 <= localization_workers <= 16:
        raise ValueError("localization_workers must be between 1 and 16.")
    if not 1 <= observation_workers <= 16:
        raise ValueError("observation_workers must be between 1 and 16.")
    raw_video = raw_video.expanduser().resolve()
    rubric_path = rubric_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache" / "observations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    assessment_image_cache_dir = output_dir / ".cache" / "assessment_images"
    localization_cache_dir = output_dir / ".cache" / "criterion_localization"
    rubric = load_rubric(rubric_path)
    observation_client_options: dict[str, Any] = {
        "api_key": observation_api_key,
        "timeout": 900.0,
    }
    if observation_base_url:
        observation_client_options["base_url"] = observation_base_url
    observation_client = OpenAI(**observation_client_options)
    assessment_client_options: dict[str, Any] = {
        "api_key": assessment_api_key,
        "timeout": 900.0,
    }
    if assessment_base_url:
        assessment_client_options["base_url"] = assessment_base_url
    assessment_client = OpenAI(**assessment_client_options)
    refinement_agent = JsonRefinementAgent(
        client=assessment_client,
        provider=assessment_provider,
        model=assessment_model,
    )

    annotated_value = perception.get("artifacts", {}).get("annotated_video")
    annotated_video = Path(annotated_value) if annotated_value else None
    observation_endpoint_ready = False
    observation_endpoint_lock = Lock()

    def analyze_event(index: int, event: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        nonlocal observation_endpoint_ready
        cache_material = json.dumps(
            {
                "event": event,
                "perception_cache_key": perception.get("cache", {}).get("key"),
                "provider": observation_provider,
                "model": observation_model,
                "rubric": rubric["rubric_id"],
                "frames": frames_per_event,
                "max_image_side": max_image_side,
                "annotated": include_annotated_frames,
                "image_detail": image_detail,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir / f"{event['event_id']}_{cache_key}.json"
        if resume and cache_path.is_file():
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"VLM: reused cached event {index + 1}/{len(perception['events'])}")
        else:
            if before_observation is not None and not observation_endpoint_ready:
                with observation_endpoint_lock:
                    if not observation_endpoint_ready:
                        before_observation()
                        observation_endpoint_ready = True
            result = _analyze_event(
                observation_client,
                provider=observation_provider,
                model=observation_model,
                raw_video=raw_video,
                annotated_video=annotated_video,
                event=event,
                rubric=rubric,
                frames_per_event=frames_per_event,
                max_image_side=max_image_side,
                include_annotated_frames=include_annotated_frames,
                max_tokens=event_max_tokens,
                retries=retries,
                image_detail=image_detail,
                refinement_agent=refinement_agent,
            )
            cache_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"VLM: completed event {index + 1}/{len(perception['events'])}, "
                f"{len(result['observations'])} observations"
            )
        return index, result

    indexed_results: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=observation_workers) as executor:
        futures = [
            executor.submit(analyze_event, index, event)
            for index, event in enumerate(perception["events"])
        ]
        for future in as_completed(futures):
            indexed_results.append(future.result())
    results = [result for _, result in sorted(indexed_results)]

    observations_path = output_dir / "observations.json"
    observations_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if before_assessment is not None:
        before_assessment()
    print(
        "=== STAGE 3A: TEXT-FIRST ASSESSMENT "
        f"({assessment_provider}, {assessment_model}) ==="
    )
    raw_judgment = _judge(
        assessment_client,
        provider=assessment_provider,
        model=assessment_model,
        rubric=rubric,
        event_results=results,
        max_tokens=judge_max_tokens,
        retries=retries,
        image_detail=image_detail,
        audio_evidence=audio_evidence,
    )
    preliminary_assessment = _normalize_assessment(
        raw_judgment,
        rubric,
        results,
        confidence_threshold,
        assessment_policy="conservative",
        audio_evidence=audio_evidence,
    )
    assessment_image_manifest = _build_assessment_image_manifest(
        mode=assessment_images,
        rubric=rubric,
        event_results=results,
        preliminary_assessment=preliminary_assessment,
        uncertainty_threshold=assessment_image_confidence_threshold,
        images_per_criterion=assessment_images_per_criterion,
        context_frames=assessment_context_frames,
        critical_criterion_images=critical_criterion_images,
        rescue_step_limit=assessment_rescue_images,
    )
    if criterion_localization and assessment_image_manifest["targets"]:
        print(
            "=== STAGE 3B-L: CRITERION-AWARE TEMPORAL LOCALIZATION "
            f"({assessment_provider}, {assessment_model}) ==="
        )
        _localize_assessment_manifest(
            assessment_client,
            provider=assessment_provider,
            model=assessment_model,
            raw_video=raw_video,
            rubric=rubric,
            manifest=assessment_image_manifest,
            cache_dir=localization_cache_dir,
            resume=resume,
            workers=localization_workers,
            coarse_seconds=localization_coarse_seconds,
            candidate_count=localization_candidates,
            search_radius=localization_search_radius,
            refinement_fps=localization_refinement_fps,
            evidence_images=localization_evidence_images,
            confidence_threshold=localization_confidence_threshold,
            max_image_side=max_image_side,
            max_tokens=judge_max_tokens,
            retries=retries,
        )
        _apply_localization_sequence_guard(assessment_image_manifest)
    if assessment_image_manifest["targets"]:
        print(
            "=== STAGE 3B: INDEPENDENT ASSESSMENT IMAGE REVIEW "
            f"(mode={assessment_images}, {assessment_provider}, {assessment_model}) ==="
        )
        visual_judgment = _run_assessment_image_reviews(
            assessment_client,
            provider=assessment_provider,
            model=assessment_model,
            base_url=assessment_base_url,
            raw_video=raw_video,
            rubric=rubric,
            manifest=assessment_image_manifest,
            cache_dir=assessment_image_cache_dir,
            resume=resume,
            workers=assessment_image_workers,
            max_image_side=max_image_side,
            max_tokens=judge_max_tokens,
            retries=retries,
            image_detail=image_detail,
            audio_evidence=audio_evidence,
        )
        raw_judgment = _fuse_step_judgments(
            raw_judgment,
            visual_judgment,
            preliminary_assessment,
            rubric,
            visual_override_confidence=assessment_visual_override_confidence,
            visual_override_margin=assessment_visual_override_margin,
        )
    else:
        assessment_image_manifest["summary"].update(
            {"completed_steps": 0, "failed_steps": 0, "cache_hits": 0}
        )
    assessment = _normalize_assessment(
        raw_judgment,
        rubric,
        results,
        confidence_threshold,
        assessment_policy=assessment_policy,
        audio_evidence=audio_evidence,
    )
    failed_visual_step_ids = {
        int(item["step_id"])
        for item in assessment_image_manifest["targets"]
        if item.get("status") == "failed"
    }
    if failed_visual_step_ids:
        for step in assessment["steps"]:
            if int(step["step_id"]) not in failed_visual_step_ids:
                continue
            step["requires_review"] = True
            step["review_reasons"].append(
                "The requested independent image review failed; this result uses "
                "the text-first assessment only."
            )
        assessment["summary"]["review_steps"] = sum(
            bool(step["requires_review"]) for step in assessment["steps"]
        )
        assessment["summary"]["requires_human_review"] = True
    assessment_images_path = output_dir / "assessment_images.json"
    assessment_images_path.write_text(
        json.dumps(assessment_image_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assessment["metadata"] = {
        "framework": {
            "name": FRAMEWORK_NAME,
            "full_name": FRAMEWORK_FULL_NAME,
        },
        "source_video": str(raw_video),
        "rubric": str(rubric_path),
        "rubric_id": rubric["rubric_id"],
        "observation": {
            "provider": observation_provider,
            "model": observation_model,
            "base_url": observation_base_url,
        },
        "assessment": {
            "provider": assessment_provider,
            "model": assessment_model,
            "base_url": assessment_base_url,
            "image_mode": assessment_images,
            "image_confidence_threshold": assessment_image_confidence_threshold,
            "visual_override_confidence": assessment_visual_override_confidence,
            "visual_override_margin": assessment_visual_override_margin,
            "images_per_criterion": assessment_images_per_criterion,
            "context_frames": assessment_context_frames,
            "critical_criterion_images": critical_criterion_images,
            "image_workers": assessment_image_workers,
            "criterion_localization": {
                "enabled": criterion_localization,
                "candidate_windows": localization_candidates,
                "coarse_interval_seconds": localization_coarse_seconds,
                "search_radius_seconds": localization_search_radius,
                "refinement_fps": localization_refinement_fps,
                "evidence_images": localization_evidence_images,
                "confidence_threshold": localization_confidence_threshold,
                "workers": localization_workers,
            },
        },
        "events": len(results),
        "frames_per_event": frames_per_event,
        "observation_workers": observation_workers,
        "include_annotated_frames": include_annotated_frames,
        "image_detail": image_detail,
        "confidence_threshold": confidence_threshold,
        "assessment_policy": assessment_policy,
        "assessment_rescue_images": assessment_rescue_images,
        "assessment_image_review": assessment_image_manifest["summary"],
        "audio": audio_evidence or {"enabled": False},
    }
    assessment_path = output_dir / "assessment.json"
    scores_path = output_dir / "scores.csv"
    assessment["artifacts"] = {
        "observations_json": str(observations_path),
        "assessment_images_json": str(assessment_images_path),
        "assessment_json": str(assessment_path),
        "scores_csv": str(scores_path),
    }
    assessment_path.write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_scores(scores_path, assessment)
    return assessment
