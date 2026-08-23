import datetime as dt
import tempfile
import unittest
from pathlib import Path

from platform_paths import artifact_stem, local_timestamp, safe_identifier, unique_directory


class PlatformPathsTest(unittest.TestCase):
    def test_identifier_is_portable_and_deterministic(self):
        self.assertEqual(safe_identifier("  PCB 缺陷 / Model A  "), "pcb-缺陷-model-a")
        self.assertEqual(safe_identifier("CON", "project"), "project")
        self.assertNotIn(" ", safe_identifier("Project with spaces"))

    def test_timestamp_and_artifact_format(self):
        moment = dt.datetime(2026, 8, 24, 1, 2, 3, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        self.assertEqual(local_timestamp(moment), "20260824-010203")
        self.assertEqual(
            artifact_stem(["PCB Project", "YOLO11n", "train"], "20260824-010203"),
            "pcb-project__yolo11n__train__20260824-010203",
        )

    def test_unique_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            (root / "run").mkdir()
            self.assertEqual(unique_directory(root, "run").name, "run__02")


if __name__ == "__main__":
    unittest.main()
