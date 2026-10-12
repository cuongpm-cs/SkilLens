"""Evaluate criterion-level assessment outputs against the examination workbook."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile


SCORE_CLASSES = (0, 1, 2)
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DOC_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
STEP_PATTERN = re.compile(r"^\s*(\d+)\s*[.．、]")


@dataclass(frozen=True)
class GoldLabels:
    """One student's 30 criterion scores loaded from the examination workbook."""

    student_name: str
    scores: dict[int, int]
    descriptions: dict[int, str]
    workbook: Path

    @property
    def total_score(self) -> int:
        return sum(self.scores.values())

    @property
    def maximum_score(self) -> int:
        return len(self.scores) * max(SCORE_CLASSES)


def _column_number(reference: str) -> int:
    match = re.match(r"[A-Z]+", reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    value = 0
    for character in match.group():
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def _row_number(reference: str) -> int:
    match = re.search(r"\d+", reference)
    if match is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    return int(match.group())


def _shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    tag = f"{{{MAIN_NS}}}t"
    return ["".join(node.text or "" for node in item.iter(tag)) for item in root]


def _first_worksheet_path(archive: ZipFile) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if sheet is None:
        raise ValueError("The workbook does not contain a worksheet.")
    relation_id = sheet.attrib.get(f"{{{DOC_REL_NS}}}id")
    relations = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relation in relations:
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Cannot resolve the workbook's first worksheet.")


def _read_first_sheet(path: Path) -> dict[tuple[int, int], str | None]:
    """Read cached XLSX cell values using only the Python standard library."""
    try:
        archive = ZipFile(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot open XLSX workbook: {path}") from error
    with archive:
        shared = _shared_strings(archive)
        worksheet = ElementTree.fromstring(
            archive.read(_first_worksheet_path(archive))
        )
    cells: dict[tuple[int, int], str | None] = {}
    for cell in worksheet.findall(f".//{{{MAIN_NS}}}c"):
        reference = cell.attrib.get("r")
        if not reference:
            continue
        cell_type = cell.attrib.get("t")
        value_node = cell.find(f"{{{MAIN_NS}}}v")
        value = value_node.text if value_node is not None else None
        if cell_type == "s" and value is not None:
            value = shared[int(value)]
        elif cell_type == "inlineStr":
            value = "".join(
                node.text or ""
                for node in cell.iter(f"{{{MAIN_NS}}}t")
            )
        cells[(_row_number(reference), _column_number(reference))] = value
    return cells


def _student_from_assessment(
    assessment: dict[str, Any],
    available_names: list[str],
    assessment_path: Path,
) -> str:
    source = assessment.get("metadata", {}).get("source_video")
    stem = Path(str(source)).stem if source else assessment_path.parent.name
    matches = [name for name in available_names if name and name in stem]
    if not matches:
        raise ValueError(
            f"Cannot infer a student name from '{stem}'. Use --student-name. "
            f"The workbook contains {len(available_names)} student columns."
        )
    return max(matches, key=len)


def load_gold_labels(
    workbook: Path,
    *,
    student_name: str | None = None,
    assessment: dict[str, Any] | None = None,
    assessment_path: Path | None = None,
) -> GoldLabels:
    """Load one student's score column from the 30-criterion workbook."""
    workbook = workbook.expanduser().resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Gold-label workbook not found: {workbook}")
    cells = _read_first_sheet(workbook)

    criterion_rows: dict[int, tuple[int, str]] = {}
    for (row, column), value in cells.items():
        if column != 1 or not value:
            continue
        match = STEP_PATTERN.match(value)
        if match:
            criterion_rows[int(match.group(1))] = (row, value)
    if sorted(criterion_rows) != list(range(1, 31)):
        raise ValueError(
            "Expected criterion IDs 1-30 in the first workbook column; found "
            f"{sorted(criterion_rows)}."
        )

    header_candidates: dict[str, tuple[int, int]] = {}
    first_criterion_row = min(row for row, _ in criterion_rows.values())
    header_row = first_criterion_row - 1
    for (row, column), value in cells.items():
        if row != header_row or column == 1 or not value:
            continue
        header_candidates[value.strip()] = (row, column)
    available_names = sorted(header_candidates)
    if student_name is None:
        if assessment is None or assessment_path is None:
            raise ValueError(
                "student_name is required when no assessment metadata is supplied."
            )
        student_name = _student_from_assessment(
            assessment, available_names, assessment_path
        )
    student_name = student_name.strip()
    if student_name not in header_candidates:
        similar = [
            name
            for name in available_names
            if student_name in name or name in student_name
        ]
        hint = f" Similar names: {similar[:5]}." if similar else ""
        raise ValueError(f"Student '{student_name}' is not in the workbook.{hint}")
    _, student_column = header_candidates[student_name]

    scores: dict[int, int] = {}
    descriptions: dict[int, str] = {}
    for step_id, (row, description) in sorted(criterion_rows.items()):
        raw_score = cells.get((row, student_column))
        if raw_score is None or not str(raw_score).strip():
            raise ValueError(
                f"Gold score is empty for student '{student_name}', step {step_id}."
            )
        try:
            score = int(float(raw_score))
        except ValueError as error:
            raise ValueError(
                f"Invalid gold score '{raw_score}' for step {step_id}."
            ) from error
        if score not in SCORE_CLASSES:
            raise ValueError(
                f"Gold score for step {step_id} must be 0, 1, or 2; got {score}."
            )
        scores[step_id] = score
        descriptions[step_id] = description
    return GoldLabels(
        student_name=student_name,
        scores=scores,
        descriptions=descriptions,
        workbook=workbook,
    )


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _quadratic_weighted_kappa(
    gold: list[int],
    predicted: list[int],
) -> float | None:
    count = len(gold)
    if count == 0:
        return None
    confusion = [[0 for _ in SCORE_CLASSES] for _ in SCORE_CLASSES]
    gold_histogram = [0 for _ in SCORE_CLASSES]
    predicted_histogram = [0 for _ in SCORE_CLASSES]
    for gold_score, predicted_score in zip(gold, predicted):
        confusion[gold_score][predicted_score] += 1
        gold_histogram[gold_score] += 1
        predicted_histogram[predicted_score] += 1
    observed = 0.0
    expected = 0.0
    maximum_distance = float((len(SCORE_CLASSES) - 1) ** 2)
    for gold_score in SCORE_CLASSES:
        for predicted_score in SCORE_CLASSES:
            weight = ((gold_score - predicted_score) ** 2) / maximum_distance
            observed += weight * confusion[gold_score][predicted_score] / count
            expected += (
                weight
                * gold_histogram[gold_score]
                * predicted_histogram[predicted_score]
                / (count * count)
            )
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def calculate_criterion_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate abstention-aware and paired criterion-level metrics."""
    gold_rows = [row for row in rows if row.get("gold_score") in SCORE_CLASSES]
    paired = [row for row in gold_rows if row.get("predicted_score") in SCORE_CLASSES]
    gold = [int(row["gold_score"]) for row in paired]
    predicted = [int(row["predicted_score"]) for row in paired]
    confusion = [[0 for _ in SCORE_CLASSES] for _ in SCORE_CLASSES]
    for gold_score, predicted_score in zip(gold, predicted):
        confusion[gold_score][predicted_score] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    present_class_f1_values: list[float] = []
    weighted_f1_sum = 0.0
    total_support = 0
    for score_class in SCORE_CLASSES:
        true_positive = confusion[score_class][score_class]
        false_positive = (
            sum(confusion[row][score_class] for row in SCORE_CLASSES)
            - true_positive
        )
        false_negative = sum(confusion[score_class]) - true_positive
        support = sum(confusion[score_class])
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[str(score_class)] = {
            "precision": _round_metric(precision),
            "recall": _round_metric(recall),
            "f1": _round_metric(f1),
            "support": support,
        }
        f1_values.append(f1)
        if support:
            present_class_f1_values.append(f1)
        weighted_f1_sum += f1 * support
        total_support += support

    absolute_errors = [
        abs(gold_score - predicted_score)
        for gold_score, predicted_score in zip(gold, predicted)
    ]
    squared_errors = [error * error for error in absolute_errors]
    exact = sum(
        gold_score == predicted_score
        for gold_score, predicted_score in zip(gold, predicted)
    )
    strict_exact = sum(
        row.get("predicted_score") == row.get("gold_score") for row in gold_rows
    )
    return {
        "gold_criteria": len(gold_rows),
        "evaluated_criteria": len(paired),
        "abstained_criteria": len(gold_rows) - len(paired),
        "coverage": _round_metric(_safe_divide(len(paired), len(gold_rows))),
        "abstention_rate": _round_metric(
            _safe_divide(len(gold_rows) - len(paired), len(gold_rows))
        ),
        "strict_exact_accuracy": _round_metric(
            _safe_divide(strict_exact, len(gold_rows))
        ),
        "selective_exact_accuracy": _round_metric(
            _safe_divide(exact, len(paired))
        ),
        "mae": _round_metric(
            _safe_divide(sum(absolute_errors), len(absolute_errors))
        ),
        "rmse": _round_metric(
            math.sqrt(sum(squared_errors) / len(squared_errors))
            if squared_errors
            else None
        ),
        "macro_f1": _round_metric(
            _safe_divide(sum(f1_values), len(f1_values)) if paired else None
        ),
        "macro_f1_present_gold_classes": _round_metric(
            _safe_divide(
                sum(present_class_f1_values),
                len(present_class_f1_values),
            )
        ),
        "weighted_f1": _round_metric(
            _safe_divide(weighted_f1_sum, total_support)
        ),
        "quadratic_weighted_kappa": _round_metric(
            _quadratic_weighted_kappa(gold, predicted)
        ),
        "confusion_matrix": {
            "labels": list(SCORE_CLASSES),
            "rows_are_gold": confusion,
        },
        "per_class": per_class,
        "paired_metric_scope": (
            "selective_exact_accuracy, MAE, RMSE, F1, and kappa use only "
            "criteria with non-null predictions; strict_exact_accuracy counts "
            "abstentions as incorrect."
        ),
    }


def evaluate_assessment(
    assessment_path: Path,
    *,
    gold_labels_path: Path,
    student_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one assessment.json against one workbook student column."""
    assessment_path = assessment_path.expanduser().resolve()
    if not assessment_path.is_file():
        raise FileNotFoundError(f"Assessment file not found: {assessment_path}")
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    if not isinstance(assessment, dict) or not isinstance(assessment.get("steps"), list):
        raise ValueError(f"Invalid assessment JSON: {assessment_path}")
    gold = load_gold_labels(
        gold_labels_path,
        student_name=student_name,
        assessment=assessment,
        assessment_path=assessment_path,
    )
    predictions = {
        item.get("step_id"): item
        for item in assessment["steps"]
        if isinstance(item, dict) and isinstance(item.get("step_id"), int)
    }
    rows: list[dict[str, Any]] = []
    for step_id, gold_score in sorted(gold.scores.items()):
        prediction = predictions.get(step_id, {})
        predicted_score = prediction.get("score")
        if isinstance(predicted_score, bool) or predicted_score not in {
            0,
            1,
            2,
            None,
        }:
            predicted_score = None
        rows.append(
            {
                "student_name": gold.student_name,
                "step_id": step_id,
                "description": prediction.get("description")
                or gold.descriptions[step_id],
                "gold_score": gold_score,
                "predicted_score": predicted_score,
                "verdict": prediction.get("verdict", "missing"),
                "confidence": prediction.get("confidence"),
                "exact_match": predicted_score == gold_score,
                "absolute_error": (
                    abs(predicted_score - gold_score)
                    if predicted_score is not None
                    else None
                ),
            }
        )
    metrics = calculate_criterion_metrics(rows)
    summary = assessment.get("summary", {})
    score_range = summary.get("score_range")
    if not (
        isinstance(score_range, list)
        and len(score_range) == 2
        and all(isinstance(value, (int, float)) for value in score_range)
    ):
        score_range = [None, None]
    gold_total = gold.total_score
    lower, upper = score_range
    official_score = summary.get("official_score")
    if isinstance(official_score, bool) or not isinstance(
        official_score, (int, float)
    ):
        official_score = None
    return {
        "student_name": gold.student_name,
        "assessment_path": str(assessment_path),
        "source_video": assessment.get("metadata", {}).get("source_video"),
        "gold_labels_path": str(gold.workbook),
        "gold_total_score": gold_total,
        "maximum_score": gold.maximum_score,
        "predicted_official_score": official_score,
        "predicted_confirmed_score": summary.get("confirmed_score"),
        "predicted_score_range": score_range,
        "gold_within_predicted_range": (
            lower <= gold_total <= upper
            if lower is not None and upper is not None
            else None
        ),
        "confirmed_score_absolute_error": (
            abs(summary["confirmed_score"] - gold_total)
            if isinstance(summary.get("confirmed_score"), (int, float))
            else None
        ),
        "official_score_absolute_error": (
            abs(official_score - gold_total)
            if official_score is not None
            else None
        ),
        "metrics": metrics,
        "steps": rows,
    }


def _write_steps_csv(path: Path, videos: list[dict[str, Any]]) -> None:
    fields = [
        "student_name",
        "step_id",
        "description",
        "gold_score",
        "predicted_score",
        "verdict",
        "confidence",
        "exact_match",
        "absolute_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for video in videos:
            writer.writerows(video["steps"])


def _write_summary_csv(path: Path, videos: list[dict[str, Any]]) -> None:
    fields = [
        "student_name",
        "gold_total_score",
        "predicted_official_score",
        "predicted_confirmed_score",
        "predicted_range_low",
        "predicted_range_high",
        "gold_within_predicted_range",
        "official_score_absolute_error",
        "coverage",
        "abstention_rate",
        "strict_exact_accuracy",
        "selective_exact_accuracy",
        "mae",
        "rmse",
        "macro_f1",
        "macro_f1_present_gold_classes",
        "weighted_f1",
        "quadratic_weighted_kappa",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for video in videos:
            metrics = video["metrics"]
            score_range = video["predicted_score_range"]
            writer.writerow(
                {
                    "student_name": video["student_name"],
                    "gold_total_score": video["gold_total_score"],
                    "predicted_official_score": video["predicted_official_score"],
                    "predicted_confirmed_score": video["predicted_confirmed_score"],
                    "predicted_range_low": score_range[0],
                    "predicted_range_high": score_range[1],
                    "gold_within_predicted_range": video[
                        "gold_within_predicted_range"
                    ],
                    "official_score_absolute_error": video[
                        "official_score_absolute_error"
                    ],
                    **{
                        key: metrics[key]
                        for key in fields
                        if key in metrics
                    },
                }
            )


def evaluate_assessments(
    assessment_paths: list[Path],
    *,
    gold_labels_path: Path,
    output_dir: Path,
    student_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate one or more assessments and write machine-readable artifacts."""
    if not assessment_paths:
        raise ValueError("At least one assessment path is required.")
    if student_name and len(assessment_paths) != 1:
        raise ValueError("--student-name can only be used with one assessment.")
    videos = [
        evaluate_assessment(
            path,
            gold_labels_path=gold_labels_path,
            student_name=student_name,
        )
        for path in assessment_paths
    ]
    all_rows = [row for video in videos for row in video["steps"]]
    aggregate_metrics = calculate_criterion_metrics(all_rows)
    range_values = [
        video["gold_within_predicted_range"]
        for video in videos
        if video["gold_within_predicted_range"] is not None
    ]
    official_pairs = [
        (video["gold_total_score"], video["predicted_official_score"])
        for video in videos
        if video["predicted_official_score"] is not None
    ]
    total_absolute_errors = [
        abs(gold - predicted) for gold, predicted in official_pairs
    ]
    total_squared_errors = [error**2 for error in total_absolute_errors]
    range_widths = [
        video["predicted_score_range"][1] - video["predicted_score_range"][0]
        for video in videos
        if all(value is not None for value in video["predicted_score_range"])
    ]
    total_score_metrics = {
        "videos": len(videos),
        "official_score_videos": len(official_pairs),
        "official_score_coverage": _round_metric(
            _safe_divide(len(official_pairs), len(videos))
        ),
        "total_score_mae": _round_metric(
            _safe_divide(sum(total_absolute_errors), len(total_absolute_errors))
        ),
        "total_score_rmse": _round_metric(
            math.sqrt(sum(total_squared_errors) / len(total_squared_errors))
            if total_squared_errors
            else None
        ),
        "score_range_evaluable_videos": len(range_values),
        "score_range_coverage": _round_metric(
            _safe_divide(sum(bool(value) for value in range_values), len(range_values))
        ),
        "mean_score_range_width": _round_metric(
            _safe_divide(sum(range_widths), len(range_widths))
        ),
    }
    aggregate = {
        "videos": len(videos),
        "criterion_metrics": aggregate_metrics,
        "total_score_metrics": total_score_metrics,
        # Retained at the top level for compatibility with earlier reports.
        "score_range_coverage": total_score_metrics["score_range_coverage"],
    }

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    steps_path = output_dir / "evaluation_steps.csv"
    summary_path = output_dir / "evaluation_summary.csv"
    result = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gold_labels_path": str(gold_labels_path.expanduser().resolve()),
        "aggregate": aggregate,
        "videos": videos,
        "limitations": [
            "The workbook provides criterion scores but no gold timestamps, "
            "bounding boxes, event boundaries, or sequence-violation labels.",
            "Paired metrics exclude null model scores; always report coverage "
            "and strict_exact_accuracy beside selective metrics.",
        ],
        "artifacts": {
            "evaluation_json": str(json_path),
            "evaluation_steps_csv": str(steps_path),
            "evaluation_summary_csv": str(summary_path),
        },
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_steps_csv(steps_path, videos)
    _write_summary_csv(summary_path, videos)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate assessment.json files against XLSX gold labels."
    )
    parser.add_argument(
        "assessments",
        nargs="+",
        type=Path,
        help="One or more assessment.json files",
    )
    parser.add_argument("--gold-labels", type=Path, required=True)
    parser.add_argument(
        "--student-name",
        help="Workbook student column; inferred from the video/result name by default",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: assessment directory for one file, ./evaluation for a batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            args.assessments[0].expanduser().resolve().parent
            if len(args.assessments) == 1
            else Path("evaluation")
        )
    result = evaluate_assessments(
        args.assessments,
        gold_labels_path=args.gold_labels,
        output_dir=output_dir,
        student_name=args.student_name,
    )
    metrics = result["aggregate"]["criterion_metrics"]
    print(f"Evaluated videos: {result['aggregate']['videos']}")
    print(f"Coverage: {metrics['coverage']}")
    print(f"Strict exact accuracy: {metrics['strict_exact_accuracy']}")
    print(f"Selective exact accuracy: {metrics['selective_exact_accuracy']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']}")
    print(f"Evaluation JSON: {result['artifacts']['evaluation_json']}")


if __name__ == "__main__":
    main()
