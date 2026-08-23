import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from annotation_exports import export_dataset, import_project_package
from annotation_store import AnnotationError, AnnotationStore


class AnnotationCollaborationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = AnnotationStore(self.root / "workspace")
        self.admin = self.store.create_user("admin", "password123", "admin")
        self.annotator = self.store.create_user("student", "password123", "annotator", self.admin)
        source = self.root / "source"
        source.mkdir()
        Image.new("RGB", (100, 80), "white").save(source / "one.jpg")
        Image.new("RGB", (120, 90), "gray").save(source / "two.png")
        self.project = self.store.create_project(self.admin, "demo", ["good", "defect"], source)

    def tearDown(self):
        self.temp.cleanup()

    def test_assignment_lock_submit_review_and_exports(self):
        self.assertEqual(self.project["imported"], 2)
        assigned = self.store.assign_items(self.admin, self.project["id"], self.annotator["id"], count=1)
        self.assertEqual(assigned, 1)
        items = self.store.list_items(self.project["id"], self.annotator)
        self.assertEqual(len(items), 1)
        item = self.store.acquire_item(items[0]["id"], self.annotator)
        saved = self.store.save_item(item["id"], self.annotator, [{
            "label": "defect", "x": 10, "y": 12, "w": 30, "h": 25,
        }], item["revision"], submit=True)
        self.assertEqual(saved["status"], "submitted")
        reviewed = self.store.review_item(item["id"], self.admin, True)
        self.assertEqual(reviewed["status"], "approved")
        read_only = self.store.acquire_item(item["id"], self.annotator)
        self.assertFalse(read_only["locked_by_me"])
        with self.assertRaises(AnnotationError) as caught:
            self.store.save_item(item["id"], self.annotator, read_only["boxes"], read_only["revision"], submit=False)
        self.assertEqual(caught.exception.status, 409)

        for export_format, expected in {
            "yolo": "data.yaml",
            "coco": "annotations/instances_default.json",
            "voc": "ImageSets/Main/train.txt",
            "labelme": "annotations/000001_one.json",
        }.items():
            path = export_dataset(self.store, self.admin, self.project["id"], export_format)
            with zipfile.ZipFile(path) as archive:
                self.assertIn(expected, archive.namelist())

    def test_project_package_moves_work_between_independent_workspaces(self):
        assigned = self.store.assign_items(self.admin, self.project["id"], self.annotator["id"], count=1)
        self.assertEqual(assigned, 1)
        item = self.store.acquire_item(self.store.list_items(self.project["id"], self.annotator)[0]["id"], self.annotator)
        self.store.save_item(item["id"], self.annotator, [{"label": "good", "x": 1, "y": 2, "w": 20, "h": 30}], item["revision"], submit=True)
        package = export_dataset(self.store, self.admin, self.project["id"], "project")

        other_store = AnnotationStore(self.root / "other_workspace")
        other_admin = other_store.create_user("owner", "password123", "admin")
        imported = import_project_package(other_store, other_admin, package)
        self.assertEqual(imported["imported"], 2)
        imported_items = other_store.list_items(imported["id"], other_admin)
        self.assertEqual(sum(item["status"] == "submitted" for item in imported_items), 1)
        self.assertEqual(sum(item["status"] == "unassigned" for item in imported_items), 1)

    def test_other_annotator_cannot_open_someone_elses_task(self):
        other = self.store.create_user("other", "password123", "annotator", self.admin)
        self.store.assign_items(self.admin, self.project["id"], self.annotator["id"], count=1)
        item_id = self.store.list_items(self.project["id"], self.annotator)[0]["id"]
        with self.assertRaises(AnnotationError) as caught:
            self.store.acquire_item(item_id, other)
        self.assertEqual(caught.exception.status, 403)

    def test_unsafe_project_package_is_rejected_and_rolled_back(self):
        package = self.root / "unsafe.matproj.zip"
        manifest = {
            "schema_version": 1,
            "kind": "myautotrain_annotation_project",
            "project": {"name": "unsafe", "task_type": "detect", "labels": ["object"]},
            "items": [{"image": "images/../escape.jpg", "boxes": []}],
        }
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("project_manifest.json", json.dumps(manifest))
        before = len(self.store.list_projects(self.admin))
        with self.assertRaises(AnnotationError):
            import_project_package(self.store, self.admin, package)
        self.assertEqual(len(self.store.list_projects(self.admin)), before)


if __name__ == "__main__":
    unittest.main()
