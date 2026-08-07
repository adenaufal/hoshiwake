from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporter import print_summary, write_csv  # noqa: E402


def make_record(**overrides) -> dict:
    record = {
        "filename": "img.png",
        "category": "SFW",
        "label": "sfw",
        "score": 0.9,
        "sfw_score": 0.9,
        "nsfw_score": 0.1,
        "all_scores": {"sfw": 0.9, "nsfw": 0.1},
        "destination": "out/SFW/img.png",
        "status": "sorted",
    }
    record.update(overrides)
    return record


class WriteCsvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_writes_all_columns(self):
        path = write_csv([make_record()], self.tmp)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["filename"], "img.png")
        self.assertEqual(row["category"], "SFW")
        self.assertEqual(row["confidence"], "0.900000")
        self.assertEqual(row["sfw_score"], "0.900000")
        self.assertEqual(row["nsfw_score"], "0.100000")
        self.assertIn('"nsfw"', row["all_scores"])
        self.assertEqual(row["destination"], "out/SFW/img.png")
        self.assertEqual(row["status"], "sorted")

    def test_handles_skipped_record_without_scores(self):
        record = make_record(
            score=None, sfw_score=None, nsfw_score=None, all_scores={}, status="skipped"
        )
        path = write_csv([record], self.tmp)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["confidence"], "")
        self.assertEqual(rows[0]["all_scores"], "")

    def test_filenames_with_commas_and_unicode_survive_roundtrip(self):
        name = 'weird, "name" 星分け.png'
        path = write_csv([make_record(filename=name)], self.tmp)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["filename"], name)


class PrintSummaryTests(unittest.TestCase):
    def summary_output(self, records) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print_summary(records)
        return buffer.getvalue()

    def test_counts_categories_skips_and_errors(self):
        records = [
            make_record(),
            make_record(category="NSFW"),
            make_record(status="skipped"),
            make_record(status="error"),
        ]
        output = self.summary_output(records)
        self.assertIn("Images processed: 4", output)
        self.assertIn("SFW:", output)
        self.assertIn("Skipped:", output)
        self.assertIn("Errors:", output)

    def test_empty_records_do_not_divide_by_zero(self):
        output = self.summary_output([])
        self.assertIn("Images processed: 0", output)


if __name__ == "__main__":
    unittest.main()
