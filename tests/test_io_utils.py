from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.io_utils import create_run_paths, load_h5ad


class IOUtilsTests(unittest.TestCase):
    def test_create_run_paths_rejects_existing_nonempty_run_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "example_run"
            run_dir.mkdir(parents=True)
            (run_dir / "existing.txt").write_text("hello", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_run_paths(
                    {
                        "output_root": root / "results",
                        "report_root": root / "reports",
                        "run_name": "example_run",
                        "overwrite_existing_run": False,
                    }
                )

    def test_create_run_paths_cleans_existing_run_when_overwrite_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "results" / "example_run"
            run_dir.mkdir(parents=True)
            stale_file = run_dir / "stale.txt"
            stale_file.write_text("hello", encoding="utf-8")

            paths = create_run_paths(
                {
                    "output_root": root / "results",
                    "report_root": root / "reports",
                    "run_name": "example_run",
                    "overwrite_existing_run": True,
                }
            )

            self.assertFalse(stale_file.exists())
            self.assertTrue(paths.summaries_dir.exists())

    def test_load_h5ad_rejects_non_h5ad_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "not_h5ad.txt"
            input_path.write_text("not an h5ad", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_h5ad(input_path)


if __name__ == "__main__":
    unittest.main()
