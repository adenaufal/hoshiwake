from __future__ import annotations

import csv
import json
from pathlib import Path

from config import CATEGORIES

FIELDNAMES = [
    "filename",
    "category",
    "label",
    "confidence",
    "sfw_score",
    "nsfw_score",
    "all_scores",
    "destination",
    "status",
]


def _format_score(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


def write_csv(records: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sort_report.csv"

    with csv_path.open("w", newline="", encoding="utf-8", errors="replace") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for record in records:
            all_scores = record.get("all_scores") or {}
            writer.writerow(
                {
                    "filename": record.get("filename", ""),
                    "category": record.get("category", ""),
                    "label": record.get("label", ""),
                    "confidence": _format_score(record.get("score")),
                    "sfw_score": _format_score(record.get("sfw_score")),
                    "nsfw_score": _format_score(record.get("nsfw_score")),
                    "all_scores": json.dumps(all_scores, sort_keys=True) if all_scores else "",
                    "destination": record.get("destination", ""),
                    "status": record.get("status", ""),
                }
            )

    return csv_path


def print_summary(records: list[dict], report_path: Path | None = None) -> None:
    total = len(records)
    category_counts = {category: 0 for category in CATEGORIES}
    skipped = 0
    errors = 0

    for record in records:
        status = record.get("status", "")
        category = record.get("category", "")
        if status == "skipped":
            skipped += 1
            continue
        if status == "error":
            errors += 1
            continue
        if category in category_counts:
            category_counts[category] += 1

    def percentage(count: int) -> float:
        if total == 0:
            return 0.0
        return (count / total) * 100

    print("\n=== Sort Summary ===")
    print(f"Images processed: {total}")
    for category in CATEGORIES:
        count = category_counts[category]
        print(f"{category + ':':<10}{count:>6} ({percentage(count):>5.1f}%)")
    print(f"{'Skipped:':<10}{skipped:>6} ({percentage(skipped):>5.1f}%)")
    if errors:
        print(f"{'Errors:':<10}{errors:>6} ({percentage(errors):>5.1f}%)")
    if report_path is not None:
        print(f"Report saved to: {report_path}")
