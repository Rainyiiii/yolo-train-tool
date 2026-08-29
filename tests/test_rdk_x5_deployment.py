import json
import tempfile
import unittest
from pathlib import Path

from rdk_x5_deployment import calibration_images, create_rdk_x5_bundle


class RdkX5DeploymentTest(unittest.TestCase):
    def test_bundle_is_portable_and_marks_bin_as_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "custom.pt"
            onnx = root / "custom.onnx"
            calibration = root / "calibration"
            output = root / "output"
            source.write_bytes(b"pt")
            onnx.write_bytes(b"onnx")
            calibration.mkdir()
            for index in range(24):
                (calibration / f"sample-{index}.jpg").write_bytes(b"image")

            result = create_rdk_x5_bundle(
                output,
                source,
                onnx,
                calibration,
                (640, 640),
                {"1": "bad", "0": "good"},
            )

            bundle = Path(result["bundle"])
            self.assertEqual(result["status"], "conversion_required")
            self.assertEqual(result["calibration_image_count"], 24)
            self.assertTrue((bundle / "model" / "trained-model.pt").is_file())
            self.assertTrue((bundle / "model" / "intermediate-model.onnx").is_file())
            self.assertEqual(len(calibration_images(bundle / "calibration_images")), 24)
            self.assertEqual((bundle / "classes.txt").read_text(encoding="utf-8"), "good\nbad\n")
            self.assertIn("ultralytics>=8.4,<9", (bundle / "requirements-rdk-x5.txt").read_text(encoding="utf-8"))

            script = (bundle / "convert_rdk_x5.sh").read_text(encoding="utf-8")
            self.assertIn("export_monkey_patch.py", script)
            self.assertIn("--branch rdk_x5", script)
            self.assertIn("hb_mapper", script)
            self.assertIn("*_bayese_*_nv12.bin", script)
            self.assertNotIn(".hbm", script)
            self.assertNotIn("rm -rf", script)

            plan = json.loads((bundle / "conversion-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["march"], "bayes-e")
            self.assertEqual(plan["final_format"], "bin")
            self.assertEqual(plan["runtime_input"], "nv12")
            self.assertEqual(plan["calibration_images"], 24)

    def test_missing_calibration_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "custom.onnx"
            source.write_bytes(b"onnx")
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(ValueError, "校准图片"):
                create_rdk_x5_bundle(root / "output", source, source, empty, 640, {})


if __name__ == "__main__":
    unittest.main()
