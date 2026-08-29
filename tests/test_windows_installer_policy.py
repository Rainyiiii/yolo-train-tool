from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = ROOT / "installer" / "windows" / "setup.iss"
RUNTIME_SCRIPT = ROOT / "install_runtime.ps1"
POWERSHELL_ENTRY_SCRIPTS = (
    ROOT / "install_and_start.ps1",
    RUNTIME_SCRIPT,
    ROOT / "installer" / "windows" / "build-installer.ps1",
)


class WindowsInstallerUninstallPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SETUP_SCRIPT.read_text(encoding="utf-8")
        cls.runtime_script = RUNTIME_SCRIPT.read_text(encoding="utf-8")

    def test_generated_program_directories_are_removed_explicitly(self) -> None:
        for directory in ("Runtime", "App", "Desktop"):
            self.assertIn(
                f'Type: filesandordirs; Name: "{{app}}\\{directory}"',
                self.script,
            )

    def test_workspace_deletion_requires_explicit_policy(self) -> None:
        self.assertIn("/PURGEDATA", self.script)
        self.assertIn("/KEEPDATA", self.script)
        self.assertIn("UninstallSilent", self.script)
        self.assertIn("WorkspaceRoot := PathCombine(InstallRoot, 'Workspace')", self.script)
        self.assertIn("DelTree(WorkspaceRoot, True, True, True)", self.script)
        self.assertIn("CurUninstallStep = usPostUninstall", self.script)

    def test_install_root_is_never_wildcard_deleted(self) -> None:
        self.assertNotIn('Name: "{app}\\*"', self.script)
        self.assertNotIn("DelTree(ExpandConstant('{app}')", self.script)
        self.assertNotIn("DelTree(InstallRoot", self.script)
        self.assertIn('Type: dirifempty; Name: "{app}"', self.script)

    def test_upgrade_defaults_to_incremental_runtime_reuse(self) -> None:
        self.assertIn("DotNetDesktopRuntimeAvailable", self.script)
        self.assertIn("WebView2RuntimeAvailable", self.script)
        self.assertIn("步骤 1 / 3 · 跳过重复安装", self.script)
        self.assertIn("步骤 2 / 3 · 跳过重复安装", self.script)
        self.assertIn(".yolo-dependency-state.json", self.runtime_script)
        self.assertIn("Test-RuntimeDependencies", self.runtime_script)
        self.assertIn("跳过依赖下载", self.runtime_script)
        self.assertIn("-m pip install -r $Requirements", self.runtime_script)
        self.assertNotIn("-m pip install --upgrade -r $Requirements", self.runtime_script)

    def test_full_runtime_repair_requires_explicit_installer_choice(self) -> None:
        self.assertIn("RepairOptionsPage.Values[0] := False", self.script)
        self.assertIn("完整修复运行环境", self.script)
        self.assertIn("RuntimeArguments := RuntimeArguments + ' -RepairRuntime'", self.script)
        self.assertIn("[switch]$RepairRuntime", self.runtime_script)
        self.assertIn("Remove-RuntimeForRepair", self.runtime_script)

    def test_windows_powershell_51_entry_scripts_use_utf8_bom(self) -> None:
        for script in POWERSHELL_ENTRY_SCRIPTS:
            with self.subTest(script=script.name):
                self.assertTrue(
                    script.read_bytes().startswith(b"\xef\xbb\xbf"),
                    f"{script.name} must keep a UTF-8 BOM for Windows PowerShell 5.1",
                )

    def test_installer_embeds_numeric_file_and_product_versions(self) -> None:
        self.assertIn("VersionInfoVersion={#ProductVersionNumeric}", self.script)
        self.assertIn("VersionInfoProductVersion={#ProductVersionNumeric}", self.script)


if __name__ == "__main__":
    unittest.main()
