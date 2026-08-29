from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from platform_subprocess import hidden_creationflags


ROOT = Path(__file__).resolve().parents[1]


class WindowsHiddenProcessTest(unittest.TestCase):
    def test_background_flags_include_no_window_and_optional_process_group(self) -> None:
        flags = hidden_creationflags(new_process_group=True)
        if os.name == "nt":
            self.assertTrue(flags & subprocess.CREATE_NO_WINDOW)
            self.assertTrue(flags & subprocess.CREATE_NEW_PROCESS_GROUP)
        else:
            self.assertEqual(flags, 0)

    @unittest.skipUnless(os.name == "nt", "Windows console behavior")
    def test_real_python_child_has_no_console_window(self) -> None:
        code = "import ctypes; print(int(ctypes.windll.kernel32.GetConsoleWindow()))"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            creationflags=hidden_creationflags(),
        )
        self.assertEqual(result.stdout.strip(), "0")

    def test_web_task_runner_and_desktop_shell_hide_child_processes(self) -> None:
        panel = (ROOT / "train_panel.py").read_text(encoding="utf-8")
        desktop = (ROOT / "desktop" / "YOLOTeamTrainingPlatform.Desktop" / "Program.cs").read_text(encoding="utf-8")
        self.assertIn("return hidden_creationflags(new_process_group=True)", panel)
        self.assertIn("CreateNoWindow = true", desktop)
        self.assertIn("WindowStyle = ProcessWindowStyle.Hidden", desktop)

    def test_source_launchers_delegate_to_windowless_vbs_entries(self) -> None:
        launchers = {
            "启动YOLO团队训练平台.cmd": "启动YOLO团队训练平台.vbs",
            "启动个人标注中心.cmd": "启动个人标注中心.vbs",
            "开启局域网协作标注.cmd": "开启局域网协作标注.vbs",
            "关闭YOLO团队训练平台.cmd": "关闭YOLO团队训练平台.vbs",
            "关闭协作标注中心.cmd": "关闭协作标注中心.vbs",
        }
        for command_name, script_name in launchers.items():
            with self.subTest(command=command_name):
                self.assertTrue((ROOT / script_name).is_file())
                command = (ROOT / command_name).read_text(encoding="utf-8-sig").casefold()
                self.assertIn("wscript.exe", command)
                self.assertIn(script_name.casefold(), command)

    def test_installer_packages_shared_subprocess_policy(self) -> None:
        build_script = (ROOT / "installer" / "windows" / "build-installer.ps1").read_text(encoding="utf-8-sig")
        team_package = (ROOT / "make_team_package.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('"platform_subprocess.py"', build_script)
        self.assertIn('".vbs"', team_package)

    def test_background_task_modules_use_shared_hidden_policy(self) -> None:
        for name in (
            "host_train_export.py",
            "rdk_x5_remote.py",
            "remote_train_env.py",
            "system_check.py",
            "panel_service.py",
            "annotation_service.py",
        ):
            with self.subTest(module=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("hidden_creationflags", source)


if __name__ == "__main__":
    unittest.main()
