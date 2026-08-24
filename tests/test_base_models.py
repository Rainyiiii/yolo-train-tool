from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from base_models import MODEL_ENTRIES, base_model_catalog, download_base_model, validate_model_name
from train_panel import HTML_PAGE, command_for, clean_values


class BaseModelCatalogTest(unittest.TestCase):
    def test_catalog_contains_supported_yolo26_detection_and_classification_models(self) -> None:
        names = {entry["name"] for entry in MODEL_ENTRIES}
        self.assertIn("yolo26n.pt", names)
        self.assertIn("yolo26x.pt", names)
        self.assertIn("yolo26n-cls.pt", names)
        self.assertEqual(len(names), len(MODEL_ENTRIES))

    def test_only_catalog_models_are_accepted(self) -> None:
        self.assertEqual(validate_model_name("yolo26n.pt")["task"], "detect")
        with self.assertRaises(ValueError):
            validate_model_name("../../untrusted.pt")
        with self.assertRaises(ValueError):
            validate_model_name("https://example.com/model.pt")

    def test_existing_model_is_reused_without_importing_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "yolo26n.pt"
            model.write_bytes(b"0" * (1024 * 1024))
            result = download_base_model("yolo26n.pt", root)
            self.assertFalse(result["downloaded"])
            self.assertEqual(Path(result["path"]), model)
            catalog = base_model_catalog(root)
            item = next(entry for entry in catalog["models"] if entry["name"] == "yolo26n.pt")
            self.assertTrue(item["downloaded"])

    def test_panel_exposes_download_page_and_background_command(self) -> None:
        self.assertIn('id="tab-models"', HTML_PAGE)
        self.assertIn("/api/base-models/download", HTML_PAGE)
        values = clean_values({})
        values["model_download_name"] = "yolo26n.pt"
        values["model_download_force"] = False
        command = [str(item) for item in command_for("model_download", values)]
        self.assertIn("base_models.py", command[1])
        self.assertIn("yolo26n.pt", command)
        self.assertNotIn("--force", command)


if __name__ == "__main__":
    unittest.main()
