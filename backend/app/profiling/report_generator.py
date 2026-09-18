"""
Data Summary Report Generator
==============================

Takes a ProfileReport (from DatasetProfiler) and writes it out as either:

  * JSON  - a single structured file, easy for a frontend/API to consume
  * CSV   - a human-readable summary report, in two sections:
              1. Overall dataset stats (row count, column count, duplicates)
              2. Per-column stats (data type, missing count/percentage)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Union

from .schema import ProfileReport


class ReportGenerator:
    """Writes a ProfileReport to disk as JSON or CSV."""

    def to_json(self, report: ProfileReport, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return path

    def to_csv(self, report: ProfileReport, output_path: Union[str, Path]) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["Dataset Summary"])
            writer.writerow(["Source File", report.source_file])
            writer.writerow(["Generated At", report.generated_at])
            writer.writerow(["Row Count", report.row_count])
            writer.writerow(["Column Count", report.column_count])
            writer.writerow(["Duplicate Row Count", report.duplicate_row_count])
            writer.writerow([])

            writer.writerow(
                ["Column Name", "Data Type", "Missing Count", "Missing Percentage"]
            )
            for col in report.columns:
                writer.writerow(
                    [col.name, col.dtype, col.missing_count, col.missing_percentage]
                )

        return path