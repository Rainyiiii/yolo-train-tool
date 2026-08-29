from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from annotation_store import AnnotationError, AnnotationStore
from annotation_ui import ANNOTATION_HTML


class AnnotationWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = AnnotationStore(self.root / "annotation")
        self.admin = self.store.create_user("admin", "password123", "admin")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _dataset(self) -> Path:
        root = self.root / "dataset"
        (root / "images").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), "white").save(root / "images" / "one.jpg")
        Image.new("RGB", (80, 60), "gray").save(root / "images" / "two.jpg")
        return root

    def test_platform_projects_sync_without_duplicates(self) -> None:
        dataset = self._dataset()
        platform_project = {
            "id": "parts",
            "name": "零件检测",
            "task": "detect",
            "labels": ["part", "defect"],
            "dataset_root": str(dataset),
        }
        first = self.store.sync_platform_projects(self.admin, [platform_project])
        second = self.store.sync_platform_projects(self.admin, [platform_project])
        projects = self.store.list_projects(self.admin)
        self.assertEqual(first, {"created": 1, "updated": 0, "imported": 2, "skipped": 0})
        self.assertEqual(second["imported"], 0)
        self.assertEqual(len(projects), 1)
        self.assertEqual(projects[0]["platform_project_id"], "parts")
        self.assertEqual(projects[0]["item_count"], 2)
        self.assertFalse(projects[0]["review_enabled"])

    def test_review_is_optional_and_disabled_by_default(self) -> None:
        project = self.store.create_project(self.admin, "简单标注", ["object"], self._dataset())
        first = self.store.acquire_item(self.store.list_items(project["id"], self.admin)[0]["id"], self.admin)
        completed = self.store.save_item(
            first["id"],
            self.admin,
            [{"label": "object", "x": 5, "y": 6, "w": 20, "h": 18}],
            first["revision"],
            submit=True,
        )
        self.assertEqual(completed["status"], "approved")

        enabled = self.store.set_project_review_mode(self.admin, project["id"], True)
        self.assertTrue(enabled["review_enabled"])
        second_id = next(item["id"] for item in self.store.list_items(project["id"], self.admin) if item["id"] != first["id"])
        second = self.store.acquire_item(second_id, self.admin)
        submitted = self.store.save_item(
            second["id"],
            self.admin,
            [{"label": "object", "x": 8, "y": 9, "w": 24, "h": 20}],
            second["revision"],
            submit=True,
        )
        self.assertEqual(submitted["status"], "submitted")

    def test_todo_queue_excludes_completed_work_and_prioritizes_resume(self) -> None:
        project = self.store.create_project(self.admin, "连续标注", ["object"], self._dataset())
        item_ids = [value["id"] for value in self.store.list_items(project["id"], self.admin)]
        first = self.store.acquire_item(item_ids[0], self.admin)
        saved = self.store.save_item(
            first["id"],
            self.admin,
            [{"label": "object", "x": 5, "y": 6, "w": 20, "h": 18}],
            first["revision"],
            submit=False,
        )
        second = self.store.acquire_item(item_ids[1], self.admin)
        self.store.save_item(second["id"], self.admin, [], second["revision"], submit=True)

        todo = self.store.list_items(project["id"], self.admin, "todo")
        self.assertEqual([value["id"] for value in todo], [saved["id"]])
        self.assertEqual(todo[0]["status"], "in_progress")

    def test_ui_exposes_crosshair_and_single_complete_action(self) -> None:
        self.assertIn("function drawCrosshair", ANNOTATION_HTML)
        self.assertIn("ctx.moveTo(0,y)", ANNOTATION_HTML)
        self.assertIn("ctx.moveTo(x,0)", ANNOTATION_HTML)
        self.assertIn("X ${x}  Y ${y}", ANNOTATION_HTML)
        self.assertIn("完成并下一张", ANNOTATION_HTML)
        self.assertIn("同步项目中心", ANNOTATION_HTML)
        self.assertNotIn(">提交审核</button>", ANNOTATION_HTML)

    def test_ui_exposes_continuous_annotation_controls(self) -> None:
        self.assertIn('<option value="todo" selected>待处理</option>', ANNOTATION_HTML)
        self.assertIn('id="saveStatus"', ANNOTATION_HTML)
        self.assertIn("function scheduleAutosave", ANNOTATION_HTML)
        self.assertIn("function completeEmptyAndNext", ANNOTATION_HTML)
        self.assertIn("function releaseCurrentItem", ANNOTATION_HTML)
        self.assertIn("空图并下一张", ANNOTATION_HTML)
        self.assertIn("event.button===1||spaceHeld", ANNOTATION_HTML)
        self.assertIn("index>=0&&!event.shiftKey", ANNOTATION_HTML)
        self.assertIn("/^[1-9]$/.test(event.key)", ANNOTATION_HTML)
        self.assertIn("tool==='select'&&selectedBox>=0", ANNOTATION_HTML)

    def test_admin_can_delete_only_the_managed_annotation_copy(self) -> None:
        dataset = self._dataset()
        project = self.store.create_project(self.admin, "待删除", ["object"], dataset)
        project_dir = self.store.projects_dir / str(project["id"])
        self.assertTrue(project_dir.is_dir())

        annotator = self.store.create_user("student", "password123", "annotator", self.admin)
        with self.assertRaises(AnnotationError) as caught:
            self.store.delete_project(annotator, project["id"])
        self.assertEqual(caught.exception.status, 403)

        deleted = self.store.delete_project(self.admin, project["id"])
        self.assertEqual(deleted["name"], "待删除")
        self.assertFalse(project_dir.exists())
        self.assertTrue(dataset.is_dir())
        self.assertEqual(self.store.list_projects(self.admin), [])

    def test_ui_exposes_guarded_project_deletion(self) -> None:
        self.assertIn('id="deleteProjectSection"', ANNOTATION_HTML)
        self.assertIn('id="annotationProjectDeleteModal"', ANNOTATION_HTML)
        self.assertIn("function openDeleteProjectDialog", ANNOTATION_HTML)
        self.assertIn("function updateDeleteProjectGuard", ANNOTATION_HTML)
        self.assertIn("function confirmDeleteCurrentProject", ANNOTATION_HTML)
        self.assertIn("'/api/projects/delete'", ANNOTATION_HTML)
        self.assertIn("输入完整项目名称确认", ANNOTATION_HTML)

    def test_project_manager_uses_progressive_disclosure(self) -> None:
        self.assertIn('id="managerContext"', ANNOTATION_HTML)
        self.assertGreaterEqual(ANNOTATION_HTML.count('class="manager-section"'), 4)
        self.assertIn("团队成员与任务", ANNOTATION_HTML)
        self.assertIn("项目包与数据集导出", ANNOTATION_HTML)


if __name__ == "__main__":
    unittest.main()
