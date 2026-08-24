import argparse
import json
import tempfile
import unittest
from pathlib import Path

from host_train_export import write_training_manifest
from model_assets import collect_model_assets, register_asset_manifest, register_asset_root


class ModelAssetsTest(unittest.TestCase):
    def test_training_manifest_groups_dataset_and_attaches_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "factory_dataset"
            images = dataset_root / "images"
            output = root / "training-runs" / "widgets" / "widgets__widget__train__20260823-120000"
            run_dir = root / "run"
            images.mkdir(parents=True)
            output.mkdir(parents=True)
            run_dir.mkdir()
            training_image = images / "sample.jpg"
            training_image.write_bytes(b"image")
            (output / "model-best.pt").write_bytes(b"pt")
            (output / "model-best.onnx").write_bytes(b"onnx")
            (output / "dataset-classes.txt").write_text("good\ndefect\n", encoding="utf-8")
            (output / "training-metrics.csv").write_text("epoch,metrics/mAP50(B)\n4,0.812\n", encoding="utf-8")
            (output / "test-evaluation").mkdir()
            args = argparse.Namespace(
                stop_export_signal="", train_task="detect", project_name="widgets",
                model_name="widget", base_model="yolo11n.pt", img_height=480, img_width=640,
                epochs=5, batch=8, workers=2, lr0=0.005, train_device="cpu", train_mode="local",
            )
            training_manifest = write_training_manifest(
                args, output, "20260823-120000", dataset_root, images, run_dir,
                training_images=[training_image],
            )
            deploy_dir = root / "deploy"
            deploy_dir.mkdir()
            deployed = deploy_dir / "widget_ncnn_model"
            deployed.mkdir()
            deployment_manifest = deploy_dir / "widget.raspberry_pi.manifest.json"
            deployment_manifest.write_text(json.dumps({
                "target": "raspberry_pi",
                "target_label": "树莓派（CPU）",
                "format": "ncnn",
                "chip": None,
                "source_model": str(output / "model-best.pt"),
                "artifact": str(deployed),
            }, ensure_ascii=False), encoding="utf-8")
            registry = root / "model-registry.json"
            register_asset_manifest(registry, training_manifest)
            register_asset_manifest(registry, deployment_manifest)

            catalog = collect_model_assets(registry, deployment_roots=[deploy_dir])
            self.assertEqual(catalog["summary"]["dataset_count"], 1)
            self.assertEqual(catalog["summary"]["run_count"], 1)
            self.assertEqual(catalog["summary"]["model_count"], 2)
            self.assertEqual(catalog["summary"]["deployment_count"], 1)
            dataset = catalog["datasets"][0]
            self.assertEqual(dataset["image_count"], 1)
            self.assertTrue(dataset["version"])
            run = dataset["runs"][0]
            self.assertEqual(run["association"], "manifest")
            self.assertEqual(run["classes"], ["good", "defect"])
            self.assertEqual(run["metrics"]["metrics/mAP50(B)"], 0.812)
            self.assertTrue(next(item for item in run["artifacts"] if item["kind"] == "test_evaluation")["exists"])
            self.assertEqual(run["deployments"][0]["format"], "ncnn")

    def test_directory_without_manifest_is_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "training-runs"
            output = root / "project" / "unmanaged-run"
            output.mkdir(parents=True)
            (output / "legacy.pt").write_bytes(b"pt")
            (output / "classes.txt").write_text("object\n", encoding="utf-8")
            registry = Path(tmp) / "model-registry.json"
            register_asset_root(registry, root)

            catalog = collect_model_assets(registry)
            self.assertEqual(catalog["summary"]["run_count"], 0)
            self.assertEqual(catalog["datasets"], [])


if __name__ == "__main__":
    unittest.main()
