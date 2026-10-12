"""Manage and validate rubric files in the knowledge base."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .knowledge_base import DEFAULT_KNOWLEDGE_BASE, RubricKnowledgeBase, normalize_rubric


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KNOWLEDGE_BASE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("validate")
    add = subparsers.add_parser("add")
    add.add_argument("rubric", type=Path)
    add.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    directory = args.knowledge_base.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "add":
        source = args.rubric.expanduser().resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        normalized = normalize_rubric(raw, source)
        destination = directory / f"{normalized['framework']['exam_id']}.json"
        if destination.exists() and not args.replace:
            raise FileExistsError(f"Rubric already exists: {destination}; use --replace")
        shutil.copy2(source, destination)
        # Re-scan after writing so duplicate aliases and cross-file conflicts are checked.
        RubricKnowledgeBase((directory,))
        print(f"Added {normalized['framework']['exam_id']}: {destination}")
        return 0
    kb = RubricKnowledgeBase((directory,))
    failed = False
    for spec in kb.list():
        warnings = kb.validate_assets(spec)
        if args.command == "list":
            print(
                f"{spec.exam_id}\t{spec.rubric['title']}\t"
                f"{len(spec.rubric['steps'])} criteria\t{spec.default_backend}"
            )
        else:
            print(f"OK {spec.exam_id}: {len(spec.rubric['steps'])} criteria")
            for warning in warnings:
                failed = True
                print(f"WARNING {spec.exam_id}: {warning}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

