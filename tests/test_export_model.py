import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from export_model import export_model, parse_imgsz


class ExportModelTest(unittest.TestCase):
    def test_existing_onnx_can_be_packaged_without_loading_ultralytics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "demo.onnx"
            model.write_bytes(b"test-onnx-placeholder")
            args = argparse.Namespace(
                model=str(model), target="generic_onnx", format="auto", chip="",
                output_dir=str(root / "deploy"), data="", int8=False,
                imgsz=parse_imgsz("480,640"),
            )
            with patch("export_model.inspect_onnx_runtime", return_value={"provider": "CPUExecutionProvider"}):
                artifact, manifest_path = export_model(args)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(artifact.is_file())
            self.assertEqual(manifest["format"], "onnx")
            self.assertEqual(manifest["input_size"], [480, 640])
            self.assertEqual(manifest["target"], "generic_onnx")

    def test_int8_requires_calibration_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "demo.onnx"
            model.write_bytes(b"placeholder")
            args = argparse.Namespace(
                model=str(model), target="generic_onnx", format="auto", chip="",
                output_dir="", data="", int8=True, imgsz=640,
            )
            with self.assertRaisesRegex(ValueError, "INT8"):
                export_model(args)


if __name__ == "__main__":
    unittest.main()
