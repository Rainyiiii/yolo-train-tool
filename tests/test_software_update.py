from __future__ import annotations

import unittest
from pathlib import Path

from scripts.release_notes import extract_release_notes
from software_update import check_for_updates, select_latest_release, semantic_version_key


ROOT = Path(__file__).resolve().parents[1]


def release(version: str, *, draft: bool = False, notes: str = "changes") -> dict:
    asset_name = f"YOLO-Team-Training-Platform-Setup-v{version}.exe"
    return {
        "tag_name": f"v{version}",
        "draft": draft,
        "prerelease": "-" in version,
        "published_at": "2026-08-30T00:00:00Z",
        "body": notes,
        "html_url": f"https://github.com/Rainyiiii/yolo-train-tool/releases/tag/v{version}",
        "assets": [{
            "name": asset_name,
            "state": "uploaded",
            "size": 62_000_000,
            "digest": "sha256:" + "a" * 64,
            "browser_download_url": f"https://github.com/Rainyiiii/yolo-train-tool/releases/download/v{version}/{asset_name}",
        }],
    }


class SoftwareUpdateTest(unittest.TestCase):
    def test_semantic_versions_order_prerelease_and_stable_correctly(self) -> None:
        self.assertGreater(semantic_version_key("3.2.19-beta"), semantic_version_key("3.2.18-beta"))
        self.assertGreater(semantic_version_key("3.2.19"), semantic_version_key("3.2.19-beta"))

    def test_latest_release_includes_prereleases_and_installer_metadata(self) -> None:
        result = select_latest_release(
            [release("3.2.18-beta"), release("3.2.20-beta", notes="new update")],
            "3.2.19-beta",
        )
        self.assertTrue(result["update_available"])
        self.assertEqual(result["latest_version"], "3.2.20-beta")
        self.assertEqual(result["notes"], "new update")
        self.assertTrue(result["download_url"].startswith("https://github.com/Rainyiiii/yolo-train-tool/releases/download/"))
        self.assertTrue(result["asset_digest"].startswith("sha256:"))

    def test_draft_and_invalid_releases_are_ignored(self) -> None:
        result = select_latest_release(
            [release("99.0.0", draft=True), {"tag_name": "nightly"}, release("3.2.19-beta")],
            "3.2.19-beta",
        )
        self.assertFalse(result["update_available"])
        self.assertEqual(result["latest_version"], "3.2.19-beta")

    def test_network_failure_is_a_nonfatal_update_result(self) -> None:
        def fail() -> list[dict]:
            raise OSError("offline")

        result = check_for_updates(force=True, fetcher=fail)
        self.assertFalse(result["ok"])
        self.assertIn("offline", result["error"])
        self.assertTrue(result["current_version"])

    def test_release_notes_are_extracted_from_matching_changelog_section(self) -> None:
        changelog = "# 更新记录\n\n## 3.2.20-beta\n\n- 新功能\n- 修复问题\n\n## 3.2.19-beta\n\n- 旧内容\n"
        notes = extract_release_notes(changelog, "v3.2.20-beta")
        self.assertIn("- 新功能", notes)
        self.assertNotIn("旧内容", notes)
        self.assertIn("默认保留工作区和用户数据", notes)

    def test_release_workflow_requires_changelog_notes(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "windows-installer.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/release_notes.py", workflow)
        self.assertIn('"--notes-file", $releaseNotes', workflow)
        self.assertNotIn('"--generate-notes"', workflow)

    def test_update_ui_api_and_desktop_install_bridge_are_connected(self) -> None:
        panel = (ROOT / "train_panel.py").read_text(encoding="utf-8")
        desktop = (ROOT / "desktop" / "YOLOTeamTrainingPlatform.Desktop" / "Program.cs").read_text(encoding="utf-8")
        self.assertIn('/api/update-check', panel)
        self.assertIn('id="update-notes"', panel)
        self.assertIn("checkSoftwareUpdate(false)", panel)
        self.assertIn('type:\'install-update\'', panel)
        self.assertIn("ValidateUpdateRequest", desktop)
        self.assertIn("requestedVersion <= runningVersion", desktop)
        self.assertIn("VerifyUpdateDigest", desktop)
        self.assertIn("VerifyUpdateVersion", desktop)

    def test_installer_packages_update_runtime(self) -> None:
        installer = (ROOT / "installer" / "windows" / "build-installer.ps1").read_text(encoding="utf-8-sig")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        updater = (ROOT / "software_update.py").read_text(encoding="utf-8")
        self.assertIn('"software_update.py"', installer)
        self.assertIn("certifi>=", requirements)
        self.assertIn("_fetch_json_with_windows_https", updater)
        self.assertIn("hidden_creationflags()", updater)


if __name__ == "__main__":
    unittest.main()
