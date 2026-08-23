from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = ROOT / "installer" / "windows" / "setup.iss"


class WindowsInstallerUninstallPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SETUP_SCRIPT.read_text(encoding="utf-8")

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


if __name__ == "__main__":
    unittest.main()
