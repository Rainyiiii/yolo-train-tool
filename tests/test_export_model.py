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
            self.assertIn("__generic-onnx__onnx__", artifact.name)
            self.assertTrue(manifest_path.name.endswith(".manifest.json"))
            self.assertEqual(manifest["kind"], "yolo_team_deployment_export")

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

    def test_rdk_x5_export_records_intermediate_and_pending_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "demo.onnx"
            model.write_bytes(b"onnx")
            calibration = root / "calibration"
            calibration.mkdir()
            for index in range(20):
                (calibration / f"sample-{index}.jpg").write_bytes(b"image")
            args = argparse.Namespace(
                model=str(model), target="drobotics_rdk_x5", format="auto", chip="x5",
                output_dir=str(root / "deploy"), data="", classes="", int8=True,
                calibration_images=str(calibration), imgsz=640,
            )
            with patch("export_model.inspect_onnx_runtime", return_value={"provider": "CPUExecutionProvider"}):
                artifact, manifest_path = export_model(args)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(artifact.is_dir())
            self.assertEqual(manifest["target"], "drobotics_rdk_x5")
            self.assertEqual(manifest["final_artifact"], "Bayes-e INT8 .bin")
            self.assertTrue(manifest["intermediate_artifact"].endswith(".onnx"))
            self.assertEqual(manifest["vendor_conversion"]["status"], "conversion_required")
            self.assertTrue(manifest["vendor_conversion"]["expected_final_artifact"].endswith("*_bayese_*_nv12.bin"))


if __name__ == "__main__":
    unittest.main()
