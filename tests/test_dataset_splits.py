import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from host_train_export import evaluate_test_split, prepare_classification_yolo, prepare_voc_yolo, split_counts, write_remote_train_script
from train_panel import DEFAULT_VALUES, HTML_PAGE, build_common_args, clean_values


class DatasetSplitTest(unittest.TestCase):
    def test_panel_exposes_and_passes_three_way_split(self):
        self.assertEqual(DEFAULT_VALUES["train_ratio_percent"], "80")
        self.assertEqual(DEFAULT_VALUES["val_ratio_percent"], "10")
        self.assertIn('id="val_ratio_percent"', HTML_PAGE)
        self.assertIn('id="split_test_value"', HTML_PAGE)
        command = build_common_args(dict(DEFAULT_VALUES), "train")
        val_index = command.index("--val-ratio-percent")
        self.assertEqual(command[val_index + 1], "10")
        migrated = clean_values({"train_ratio_percent": "95"})
        self.assertEqual(migrated["train_ratio_percent"], "95")
        self.assertEqual(migrated["val_ratio_percent"], "5")

    def test_split_counts_default_to_80_10_10(self):
        self.assertEqual(split_counts(10), {"train": 8, "val": 1, "test": 1})
        self.assertEqual(split_counts(10, 80, 20), {"train": 8, "val": 2, "test": 0})
        with self.assertRaisesRegex(SystemExit, "至少需要 3 张"):
            split_counts(2)
        with self.assertRaisesRegex(SystemExit, "不能超过 100%"):
            split_counts(10, 90, 20)

    def test_detection_dataset_writes_test_split_to_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images_dir = root / "images"
            annotations_dir = root / "annotations"
            output_dir = root / "prepared"
            images_dir.mkdir()
            annotations_dir.mkdir()

            for index in range(10):
                name = f"sample-{index:02d}.jpg"
                Image.new("RGB", (64, 64), (index, 20, 30)).save(images_dir / name)
                annotation = ET.Element("annotation")
                ET.SubElement(annotation, "filename").text = name
                size = ET.SubElement(annotation, "size")
                ET.SubElement(size, "width").text = "64"
                ET.SubElement(size, "height").text = "64"
                ET.SubElement(size, "depth").text = "3"
                obj = ET.SubElement(annotation, "object")
                ET.SubElement(obj, "name").text = "object"
                box = ET.SubElement(obj, "bndbox")
                for key, value in (("xmin", "8"), ("ymin", "8"), ("xmax", "48"), ("ymax", "48")):
                    ET.SubElement(box, key).text = value
                ET.ElementTree(annotation).write(annotations_dir / f"sample-{index:02d}.xml", encoding="utf-8")

            yaml_path, _ = prepare_voc_yolo(
                images_dir,
                annotations_dir,
                output_dir,
                train_ratio_percent=80,
                val_ratio_percent=10,
                img_width=64,
                img_height=64,
            )

            self.assertEqual(len(list((output_dir / "images" / "train").glob("*.jpg"))), 8)
            self.assertEqual(len(list((output_dir / "images" / "val").glob("*.jpg"))), 1)
            self.assertEqual(len(list((output_dir / "images" / "test").glob("*.jpg"))), 1)
            self.assertIn("test: images/test", yaml_path.read_text(encoding="utf-8"))

    def test_classification_dataset_splits_every_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "prepared"
            for class_index, class_name in enumerate(("normal", "defect")):
                class_dir = source_dir / class_name
                class_dir.mkdir(parents=True)
                for image_index in range(10):
                    Image.new("RGB", (32, 32), (class_index * 80, image_index, 40)).save(class_dir / f"{image_index:02d}.jpg")

            prepare_classification_yolo(source_dir, output_dir, train_ratio_percent=80, val_ratio_percent=10)

            for class_name in ("normal", "defect"):
                self.assertEqual(len(list((output_dir / "train" / class_name).glob("*.jpg"))), 8)
                self.assertEqual(len(list((output_dir / "val" / class_name).glob("*.jpg"))), 1)
                self.assertEqual(len(list((output_dir / "test" / class_name).glob("*.jpg"))), 1)

    def test_final_evaluation_uses_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_yaml = root / "dataset.yaml"
            dataset_yaml.write_text("train: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
            best_pt = root / "best.pt"
            best_pt.write_bytes(b"test model")
            args = SimpleNamespace(
                train_task="detect", conda_env="", img_width=640, img_height=480,
                batch=8, workers=2, train_device="cpu",
            )
            with patch("host_train_export.build_env_cmd", return_value=["yolo"]), patch("host_train_export.run") as run_mock:
                result = evaluate_test_split(args, dataset_yaml, best_pt, root / "work")

            command = run_mock.call_args.args[0]
            self.assertIn("split=test", command)
            self.assertIn(f"model={best_pt}", command)
            self.assertEqual(result, root / "work" / "test-evaluation")

    def test_remote_training_also_evaluates_test_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "remote-train.ps1"
            write_remote_train_script(script_path)
            script = script_path.read_text(encoding="utf-8")
            self.assertIn('"split=test"', script)
            self.assertIn('"name=test-evaluation"', script)
            powershell = shutil.which("powershell.exe")
            if powershell:
                safe_path = str(script_path).replace("'", "''")
                parser = f"$tokens=$null;$errors=$null;[Management.Automation.Language.Parser]::ParseFile('{safe_path}',[ref]$tokens,[ref]$errors)|Out-Null;if($errors.Count){{$errors|ForEach-Object{{Write-Error $_}};exit 1}}"
                result = subprocess.run([powershell, "-NoProfile", "-Command", parser], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
