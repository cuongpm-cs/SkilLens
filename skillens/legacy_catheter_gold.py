"""Evaluate catheterization criterion scores against an examiner XLSX report."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


@dataclass(frozen=True)
class GoldLabels:
    student_name: str
    evaluator_name: str
    exam_status: str
    scores: dict[int, int]
    descriptions: dict[int, str]
    reported_total: int
    workbook: Path
    row_number: int

    @property
    def total_score(self) -> int:
        return sum(self.scores.values())


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
    try:
        archive = ZipFile(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot open XLSX workbook: {path}") from error
    with archive:
        shared = _shared_strings(archive)
        worksheet = ElementTree.fromstring(archive.read(_first_worksheet_path(archive)))
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
            value = "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
        cells[(_row_number(reference), _column_number(reference))] = value
    return cells


def _normalize_name(value: str) -> str:
    # Reports may separate name characters with punctuation while videos do not.
    return re.sub(r"[\s,，;；]", "", value).casefold()


def _infer_student_name(assessment_path: Path) -> str | None:
    config_path = assessment_path.parent / "client_config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return Path(str(config.get("video", ""))).stem or None


def _parse_score(value: str | None, *, context: str) -> int:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing score: {context}")
    try:
        score = int(float(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid score {value!r}: {context}") from error
    if score not in SCORE_CLASSES:
        raise ValueError(f"Score must be 0, 1, or 2; got {score}: {context}")
    return score


def load_gold_labels(workbook: Path, student_name: str) -> GoldLabels:
    """Select the one completed examiner row for a named candidate."""
    workbook = workbook.expanduser().resolve()
    if not workbook.is_file():
        raise FileNotFoundError(f"Gold-label workbook not found: {workbook}")
    cells = _read_first_sheet(workbook)
    if cells.get((1, 1)) != "Evaluatee Name":
        raise ValueError("Unexpected workbook layout: A1 is not 'Evaluatee Name'.")
    descriptions = {
        criterion_id: str(cells.get((1, criterion_id + 7)) or "")
        for criterion_id in range(1, 21)
    }
    target = _normalize_name(student_name)
    matching_rows = [
        row
        for row in sorted({row for row, column in cells if column == 1 and row >= 4})
        if target in _normalize_name(str(cells.get((row, 1)) or ""))
        or _normalize_name(str(cells.get((row, 1)) or "")) in target
    ]
    if not matching_rows:
        available = sorted(
            str(value)
            for (row, column), value in cells.items()
            if column == 1 and row >= 4 and value and not str(value).startswith("Download")
        )
        raise ValueError(
            f"Student {student_name!r} is not in {workbook.name}. "
            f"Available names: {available}"
        )
    completed = [
        row
        for row in matching_rows
        if str(cells.get((row, 5)) or "").strip().casefold() != "in progress"
    ]
    if len(completed) != 1:
        raise ValueError(
            f"Expected exactly one completed row for {student_name!r}; "
            f"matching rows={matching_rows}, completed rows={completed}."
        )
    row = completed[0]
    scores = {
        criterion_id: _parse_score(
            cells.get((row, criterion_id + 7)),
            context=f"row {row}, criterion {criterion_id}",
        )
        for criterion_id in range(1, 21)
    }
    reported_total = int(float(str(cells.get((row, 6)) or "nan")))
    if sum(scores.values()) != reported_total:
        raise ValueError(
            f"Workbook total mismatch at row {row}: F={reported_total}, "
            f"criterion sum={sum(scores.values())}."
        )
    return GoldLabels(
        student_name=str(cells[(row, 1)]),
        evaluator_name=str(cells.get((row, 3)) or ""),
        exam_status=str(cells.get((row, 5)) or ""),
        scores=scores,
        descriptions=descriptions,
        reported_total=reported_total,
        workbook=workbook,
        row_number=row,
    )


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _quadratic_weighted_kappa(gold: list[int], predicted: list[int]) -> float | None:
    count = len(gold)
    if not count:
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
    """Use the same abstention-aware 0/1/2 metrics as the Aseptic pipeline."""
    gold_rows = [row for row in rows if row.get("gold_score") in SCORE_CLASSES]
    paired = [row for row in gold_rows if row.get("predicted_score") in SCORE_CLASSES]
    gold = [int(row["gold_score"]) for row in paired]
    predicted = [int(row["predicted_score"]) for row in paired]
    confusion = [[0 for _ in SCORE_CLASSES] for _ in SCORE_CLASSES]
    for gold_score, predicted_score in zip(gold, predicted):
        confusion[gold_score][predicted_score] += 1

    per_class: dict[str, dict[str, float | int | None]] = {}
    all_f1: list[float] = []
    present_f1: list[float] = []
    weighted_f1_sum = 0.0
    total_support = 0
    for score_class in SCORE_CLASSES:
        true_positive = confusion[score_class][score_class]
        false_positive = sum(confusion[row][score_class] for row in SCORE_CLASSES) - true_positive
        false_negative = sum(confusion[score_class]) - true_positive
        support = sum(confusion[score_class])
        precision = _safe_divide(true_positive, true_positive + false_positive) or 0.0
        recall = _safe_divide(true_positive, true_positive + false_negative) or 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(score_class)] = {
            "precision": _round_metric(precision),
            "recall": _round_metric(recall),
            "f1": _round_metric(f1),
            "support": support,
        }
        all_f1.append(f1)
        if support:
            present_f1.append(f1)
        weighted_f1_sum += f1 * support
        total_support += support

    absolute_errors = [abs(a - b) for a, b in zip(gold, predicted)]
    squared_errors = [value * value for value in absolute_errors]
    exact = sum(a == b for a, b in zip(gold, predicted))
    soft_credits = [max(0.0, 1.0 - abs(a - b) / 2.0) for a, b in zip(gold, predicted)]
    strict_soft_credit = sum(
        max(
            0.0,
            1.0
            - abs(int(row["predicted_score"]) - int(row["gold_score"])) / 2.0,
        )
        if row.get("predicted_score") in SCORE_CLASSES
        else 0.0
        for row in gold_rows
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
        "strict_exact_accuracy": _round_metric(_safe_divide(strict_exact, len(gold_rows))),
        "selective_exact_accuracy": _round_metric(_safe_divide(exact, len(paired))),
        "strict_soft_accuracy": _round_metric(
            _safe_divide(strict_soft_credit, len(gold_rows))
        ),
        "selective_soft_accuracy": _round_metric(
            _safe_divide(sum(soft_credits), len(soft_credits))
        ),
        "mae": _round_metric(_safe_divide(sum(absolute_errors), len(absolute_errors))),
        "rmse": _round_metric(
            math.sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else None
        ),
        "macro_f1": _round_metric(_safe_divide(sum(all_f1), len(all_f1)) if paired else None),
        "macro_f1_present_gold_classes": _round_metric(
            _safe_divide(sum(present_f1), len(present_f1))
        ),
        "weighted_f1": _round_metric(_safe_divide(weighted_f1_sum, total_support)),
        "quadratic_weighted_kappa": _round_metric(
            _quadratic_weighted_kappa(gold, predicted)
        ),
        "confusion_matrix": {"labels": list(SCORE_CLASSES), "rows_are_gold": confusion},
        "per_class": per_class,
        "paired_metric_scope": (
            "Soft accuracy gives 1.0 for an exact score, 0.5 for a one-point "
            "difference, and 0.0 for a two-point difference. Selective metrics use "
            "non-null pairs; strict metrics count abstentions as zero credit."
        ),
    }


def evaluate_assessment(
    assessment_path: Path,
    *,
    gold_labels_path: Path,
    student_name: str | None = None,
) -> dict[str, Any]:
    assessment_path = assessment_path.expanduser().resolve()
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    criteria = assessment.get("criteria")
    if not isinstance(criteria, list):
        raise ValueError("assessment.json must contain a criteria list.")
    inferred_name = student_name or _infer_student_name(assessment_path)
    if not inferred_name:
        raise ValueError("Cannot infer the candidate name; use --student-name.")
    gold = load_gold_labels(gold_labels_path, inferred_name)
    predictions = {
        int(item["criterion_id"]): item
        for item in criteria
        if isinstance(item, dict) and "criterion_id" in item
    }
    rows: list[dict[str, Any]] = []
    for criterion_id in range(1, 21):
        prediction = predictions.get(criterion_id, {})
        predicted_score = prediction.get("score")
        if isinstance(predicted_score, bool) or predicted_score not in {0, 1, 2, None}:
            predicted_score = None
        imputed_score = prediction.get("imputed_score")
        if isinstance(imputed_score, bool) or imputed_score not in {0, 1, 2, None}:
            imputed_score = None
        model_score = prediction.get("model_score")
        if isinstance(model_score, bool) or model_score not in {0, 1, 2, None}:
            model_score = None
        benefit_of_doubt_score = prediction.get("benefit_of_doubt_score")
        if (
            isinstance(benefit_of_doubt_score, bool)
            or benefit_of_doubt_score not in {0, 1, 2}
        ):
            benefit_of_doubt_score = (
                predicted_score if predicted_score is not None else 2
            )
        gold_score = gold.scores[criterion_id]
        rows.append(
            {
                "student_name": gold.student_name,
                "criterion_id": criterion_id,
                "description": gold.descriptions[criterion_id],
                "gold_score": gold_score,
                "predicted_score": predicted_score,
                "model_score": model_score,
                "imputed_score": imputed_score,
                "benefit_of_doubt_score": benefit_of_doubt_score,
                "verdict": prediction.get("verdict", "missing"),
                "confidence": prediction.get("confidence"),
                "score_basis": prediction.get("score_basis"),
                "review_required": prediction.get("review_required"),
                "exact_match": predicted_score == gold_score,
                "signed_error": (
                    predicted_score - gold_score if predicted_score is not None else None
                ),
                "absolute_error": (
                    abs(predicted_score - gold_score) if predicted_score is not None else None
                ),
                "soft_credit": (
                    max(0.0, 1.0 - abs(predicted_score - gold_score) / 2.0)
                    if predicted_score is not None
                    else 0.0
                ),
                "imputed_exact_match": imputed_score == gold_score,
                "imputed_soft_credit": (
                    max(0.0, 1.0 - abs(imputed_score - gold_score) / 2.0)
                    if imputed_score is not None
                    else 0.0
                ),
            }
        )
    metrics = calculate_criterion_metrics(rows)
    imputed_rows = [
        {**row, "predicted_score": row.get("imputed_score")} for row in rows
    ]
    imputed_metrics = calculate_criterion_metrics(imputed_rows)
    model_rows = [
        {**row, "predicted_score": row.get("model_score")} for row in rows
    ]
    model_metrics = calculate_criterion_metrics(model_rows)
    benefit_rows = [
        {**row, "predicted_score": row.get("benefit_of_doubt_score")} for row in rows
    ]
    benefit_of_doubt_metrics = calculate_criterion_metrics(benefit_rows)
    raw_score = assessment.get("raw_score")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raw_score = None
    score_range = assessment.get("score_range", [None, None])
    if not isinstance(score_range, list) or len(score_range) != 2:
        score_range = [None, None]
    model_score_total = assessment.get("model_score_total")
    if isinstance(model_score_total, bool) or not isinstance(
        model_score_total, (int, float)
    ):
        model_values = [row["model_score"] for row in rows]
        model_score_total = (
            sum(model_values) if all(value is not None for value in model_values) else None
        )
    return {
        "student_name": gold.student_name,
        "evaluator_name": gold.evaluator_name,
        "exam_status": gold.exam_status,
        "gold_workbook_row": gold.row_number,
        "assessment_path": str(assessment_path),
        "assessment_sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
        "gold_labels_path": str(gold.workbook),
        "gold_total_score": gold.total_score,
        "maximum_score": 40,
        "predicted_official_score": raw_score,
        "predicted_model_score": model_score_total,
        "predicted_lenient_imputed_score": assessment.get("lenient_imputed_score"),
        "predicted_benefit_of_doubt_score": assessment.get(
            "benefit_of_doubt_score", score_range[1]
        ),
        "predicted_score_range": score_range,
        "official_score_signed_error": raw_score - gold.total_score if raw_score is not None else None,
        "official_score_absolute_error": abs(raw_score - gold.total_score) if raw_score is not None else None,
        "gold_within_predicted_range": (
            score_range[0] <= gold.total_score <= score_range[1]
            if all(isinstance(value, (int, float)) for value in score_range)
            else None
        ),
        "metrics": metrics,
        "model_metrics": model_metrics,
        "imputed_metrics": imputed_metrics,
        "benefit_of_doubt_metrics": benefit_of_doubt_metrics,
        "criteria": rows,
    }


def _write_criteria_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "student_name", "criterion_id", "description", "gold_score",
        "predicted_score", "model_score", "imputed_score", "benefit_of_doubt_score",
        "verdict", "confidence", "score_basis",
        "review_required", "exact_match", "signed_error", "absolute_error",
        "soft_credit", "imputed_exact_match", "imputed_soft_credit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    row = {
        "student_name": result["student_name"],
        "gold_total_score": result["gold_total_score"],
        "predicted_official_score": result["predicted_official_score"],
        "predicted_model_score": result["predicted_model_score"],
        "predicted_lenient_imputed_score": result["predicted_lenient_imputed_score"],
        "predicted_benefit_of_doubt_score": result[
            "predicted_benefit_of_doubt_score"
        ],
        "official_score_signed_error": result["official_score_signed_error"],
        "official_score_absolute_error": result["official_score_absolute_error"],
        **{key: value for key, value in metrics.items() if not isinstance(value, dict)},
        **{
            f"model_{key}": value
            for key, value in result["model_metrics"].items()
            if not isinstance(value, dict)
        },
        **{
            f"imputed_{key}": value
            for key, value in result["imputed_metrics"].items()
            if not isinstance(value, dict)
        },
        **{
            f"benefit_of_doubt_{key}": value
            for key, value in result["benefit_of_doubt_metrics"].items()
            if not isinstance(value, dict)
        },
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def evaluate_and_write(
    assessment_path: Path,
    *,
    gold_labels_path: Path,
    output_dir: Path,
    student_name: str | None = None,
) -> dict[str, Any]:
    video = evaluate_assessment(
        assessment_path,
        gold_labels_path=gold_labels_path,
        student_name=student_name,
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation.json"
    criteria_path = output_dir / "evaluation_criteria.csv"
    summary_path = output_dir / "evaluation_summary.csv"
    result = {
        "schema_version": "1.0",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        **video,
        "limitations": [
            "This is a single-video evaluation with only 20 criterion samples; it is not a dataset-level performance estimate.",
            "The workbook contains criterion scores but no gold timestamps, boxes, event boundaries, or evidence annotations.",
            "The completed examiner row is used; rows marked In Progress are excluded.",
        ],
        "artifacts": {
            "evaluation_json": str(json_path),
            "evaluation_criteria_csv": str(criteria_path),
            "evaluation_summary_csv": str(summary_path),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_criteria_csv(criteria_path, result["criteria"])
    _write_summary_csv(summary_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate catheterization assessment.json against an XLSX examiner row."
    )
    parser.add_argument("assessment", type=Path)
    parser.add_argument("--gold-labels", type=Path, required=True)
    parser.add_argument("--student-name")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = args.output_dir or args.assessment.expanduser().resolve().parent / "evaluation"
    result = evaluate_and_write(
        args.assessment,
        gold_labels_path=args.gold_labels,
        output_dir=output_dir,
        student_name=args.student_name,
    )
    metrics = result["metrics"]
    print(f"Candidate: {result['student_name']}")
    print(
        f"Total score: predicted={result['predicted_official_score']}/40, "
        f"gold={result['gold_total_score']}/40, "
        f"absolute_error={result['official_score_absolute_error']}"
    )
    print(f"Coverage: {metrics['coverage']}")
    print(f"Exact accuracy: {metrics['strict_exact_accuracy']}")
    print(f"Soft accuracy: {metrics['strict_soft_accuracy']}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"Macro F1: {metrics['macro_f1']}")
    print(f"Weighted F1: {metrics['weighted_f1']}")
    print(f"Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']}")
    imputed = result["imputed_metrics"]
    print(
        "Raw model: "
        f"score={result['predicted_model_score']}/40, "
        f"exact_accuracy={result['model_metrics']['strict_exact_accuracy']}, "
        f"soft_accuracy={result['model_metrics']['strict_soft_accuracy']}"
    )
    print(
        "Lenient imputed: "
        f"score={result['predicted_lenient_imputed_score']}/40, "
        f"exact_accuracy={imputed['strict_exact_accuracy']}, "
        f"soft_accuracy={imputed['strict_soft_accuracy']}"
    )
    print(
        "Benefit-of-doubt upper bound: "
        f"score={result['predicted_benefit_of_doubt_score']}/40, "
        f"exact_accuracy={result['benefit_of_doubt_metrics']['strict_exact_accuracy']}, "
        f"soft_accuracy={result['benefit_of_doubt_metrics']['strict_soft_accuracy']}"
    )
    print(f"Evaluation JSON: {result['artifacts']['evaluation_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
