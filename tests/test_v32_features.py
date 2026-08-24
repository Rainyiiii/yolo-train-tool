from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from annotation_store import AnnotationStore
from model_assets import collect_model_assets, register_external_model
from project_manager import activate_project, create_project, delete_project, duplicate_project, inspect_dataset, project_catalog, update_project


def image_bytes(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


class Version32FeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_project_registry_and_dataset_health(self) -> None:
        registry = self.root / "config" / "projects.json"
        dataset = self.root / "dataset"
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels").mkdir(parents=True)
        Image.new("RGB", (40, 30), "blue").save(dataset / "images" / "ready.jpg")
        Image.new("RGB", (40, 30), "green").save(dataset / "images" / "missing.jpg")
        (dataset / "labels" / "ready.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        with patch("project_manager.DATASETS_DIR", self.root / "managed"):
            project = create_project("零件检测", "detect", ["part"], dataset, registry)
            updated = update_project(project["id"], {"notes": "v3.2"}, registry)
            active = activate_project(project["id"], registry)
            catalog = project_catalog(registry)
        self.assertEqual(updated["notes"], "v3.2")
        self.assertEqual(active["id"], project["id"])
        self.assertEqual(catalog["active_project_id"], project["id"])
        health = inspect_dataset(dataset)
        self.assertEqual(health["image_count"], 2)
        self.assertEqual(health["box_count"], 1)
        self.assertEqual(health["missing_labels"], 1)
        self.assertEqual(health["health"], "warning")

    def test_project_lifecycle_keeps_external_data_and_can_remove_managed_data(self) -> None:
        registry = self.root / "config" / "projects.json"
        managed = self.root / "managed"
        external = self.root / "external-dataset"
        external.mkdir()
        (external / "keep.txt").write_text("user data", encoding="utf-8")
        with patch("project_manager.DATASETS_DIR", managed):
            external_project = create_project("外部数据", dataset_root=external, path=registry)
            managed_project = duplicate_project(external_project["id"], path=registry)
            self.assertNotEqual(managed_project["id"], external_project["id"])
            self.assertEqual(managed_project["labels"], external_project["labels"])
            managed_root = Path(managed_project["root"])
            (managed_root / "images" / "sample.txt").write_text("managed", encoding="utf-8")

            removed_external = delete_project(external_project["id"], path=registry)
            self.assertFalse(removed_external["deleted_managed_data"])
            self.assertTrue((external / "keep.txt").is_file())

            removed_managed = delete_project(managed_project["id"], delete_managed_data=True, path=registry)
            self.assertTrue(removed_managed["deleted_managed_data"])
            self.assertFalse(managed_root.exists())
            self.assertEqual(project_catalog(registry)["summary"]["project_count"], 0)

    def test_project_delete_rejects_a_tampered_managed_root(self) -> None:
        registry = self.root / "config" / "projects.json"
        managed = self.root / "managed"
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("do not delete", encoding="utf-8")
        with patch("project_manager.DATASETS_DIR", managed):
            project = create_project("安全边界", path=registry)
            payload = json.loads(registry.read_text(encoding="utf-8"))
            payload["projects"][0]["root"] = str(outside)
            registry.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "托管规则"):
                delete_project(project["id"], delete_managed_data=True, path=registry)
        self.assertTrue((outside / "keep.txt").is_file())

    def test_existing_model_can_be_registered_and_grouped(self) -> None:
        registry = self.root / "model-registry.json"
        model = self.root / "best.onnx"
        model.write_bytes(b"test-model")
        record = register_external_model(
            registry,
            model,
            dataset_name="零件数据集",
            dataset_root=self.root / "dataset",
            task="detect",
            project_id="parts",
            labels=["part"],
        )
        catalog = collect_model_assets(registry)
        self.assertEqual(record["path"], str(model.resolve()))
        self.assertEqual(catalog["summary"]["model_count"], 1)
        run = catalog["datasets"][0]["runs"][0]
        self.assertEqual(run["association"], "manual")
        self.assertEqual(run["classes"], ["part"])

    def test_browser_upload_imports_image_without_shared_disk_path(self) -> None:
        store = AnnotationStore(self.root / "workspace")
        admin = store.create_user("admin", "password123", "admin")
        project = store.create_project(admin, "浏览器上传", ["object"])
        uploaded = store.import_uploaded_image(admin, project["id"], "batch/camera-01.png", image_bytes())
        items = store.list_items(project["id"], admin)
        self.assertEqual(uploaded["relative_source"], "batch/camera-01.png")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["width"], 32)
        self.assertEqual(items[0]["height"], 24)


if __name__ == "__main__":
    unittest.main()
