import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rdk_x5_remote


class RdkX5RemoteTest(unittest.TestCase):
    @patch("rdk_x5_remote.subprocess.Popen")
    def test_stream_runner_decodes_mixed_wsl_output(self, popen):
        process = popen.return_value
        process.stdin = None
        process.stdout.readline.side_effect = ["WSL 警告\r\n".encode("utf-16-le"), "工具链就绪\n".encode(), b""]
        process.wait.return_value = 0
        with patch("builtins.print") as printer:
            rdk_x5_remote._run_stream(["wsl.exe"])
        rendered = [str(call.args[0]) for call in printer.call_args_list]
        self.assertIn("WSL 警告", rendered)
        self.assertIn("工具链就绪", rendered)

    def test_wsl_script_is_sent_over_stdin(self):
        command = rdk_x5_remote.build_wsl_command("Ubuntu-22.04")
        self.assertEqual(command[-2:], ["bash", "-s"])
        root_command = rdk_x5_remote.build_wsl_command("Ubuntu-22.04", "root")
        self.assertEqual(root_command[3:5], ["-u", "root"])
        script = rdk_x5_remote.wsl_probe_script(rdk_x5_remote.DEFAULT_VENV)
        self.assertIn('test "$ARCH" = "x86_64"', script)
        self.assertIn("RDK_WSL_READY=1", script)
        self.assertNotIn(script, command)

    @patch("rdk_x5_remote.shutil.which", return_value=None)
    def test_environment_status_is_optional_when_wsl_is_missing(self, _which):
        status = rdk_x5_remote.inspect_wsl_environment()
        self.assertEqual(status["overall"], "not_installed")
        self.assertTrue(status["can_install"])
        self.assertFalse(status["can_remove"])
        self.assertEqual(status["components"][0]["status"], "missing")

    @patch("rdk_x5_remote._capture_wsl")
    @patch("rdk_x5_remote.installed_wsl_distros", return_value=["Ubuntu-22.04"])
    @patch("rdk_x5_remote._capture", return_value="WSL version: 2.7.12.0")
    @patch("rdk_x5_remote.shutil.which", return_value="C:/Windows/System32/wsl.exe")
    def test_environment_status_reports_all_ready_layers(self, _which, _capture, _distros, capture_wsl):
        capture_wsl.return_value = "\n".join((
            "ARCH=x86_64",
            "SYSTEM=Ubuntu 22.04.5 LTS",
            "PYTHON=Python 3.10.12",
            "PYTHON_OK=1",
            "TOOLCHAIN_OK=1",
            "MAPPER=hb_mapper 1.0.0",
            "SIZE=2.7G",
            "FREE=120G",
        ))
        status = rdk_x5_remote.inspect_wsl_environment()
        self.assertEqual(status["overall"], "ready")
        self.assertTrue(status["can_remove"])
        self.assertFalse(status["can_install"])
        self.assertTrue(all(item["status"] == "ready" for item in status["components"]))

    @patch("rdk_x5_remote._capture_wsl", side_effect=RuntimeError("distro start failed"))
    @patch("rdk_x5_remote.installed_wsl_distros", return_value=["Ubuntu-22.04"])
    @patch("rdk_x5_remote._capture", side_effect=[RuntimeError("old WSL has no --version"), "默认版本: 2"])
    @patch("rdk_x5_remote.shutil.which", return_value="C:/Windows/System32/wsl.exe")
    def test_environment_status_falls_back_to_wsl_status_on_older_windows(self, _which, _capture, _distros, _probe):
        status = rdk_x5_remote.inspect_wsl_environment()
        self.assertEqual(status["components"][0]["status"], "ready")
        self.assertEqual(status["components"][0]["detail"], "默认版本: 2")
        self.assertEqual(status["overall"], "error")

    def test_setup_and_removal_are_guarded_by_platform_marker(self):
        setup = rdk_x5_remote.wsl_setup_script(rdk_x5_remote.DEFAULT_VENV)
        removal = rdk_x5_remote.wsl_remove_script(rdk_x5_remote.DEFAULT_VENV)
        self.assertIn(rdk_x5_remote.TOOLCHAIN_MARKER, setup)
        self.assertIn(rdk_x5_remote.TOOLCHAIN_MARKER, removal)
        self.assertIn('case "$VENV_REAL" in "$HOME_REAL"/*)', removal)
        self.assertIn("rdk-model-zoo-x5", removal)
        self.assertIn("rdk-jobs", removal)
        self.assertNotIn("--unregister", removal)

    @patch("rdk_x5_remote.inspect_wsl_environment", return_value={"overall": "ready"})
    @patch("rdk_x5_remote.installed_wsl_distros", return_value=["Ubuntu-22.04"])
    @patch("rdk_x5_remote._run_stream")
    @patch("rdk_x5_remote._require_program", return_value="wsl.exe")
    def test_install_skips_existing_distribution(self, _program, run_stream, _distros, _inspect):
        status = rdk_x5_remote.install_wsl_distro("Ubuntu-22.04")
        self.assertEqual(status["overall"], "ready")
        run_stream.assert_not_called()

    def test_linux_paths_and_host_reject_shell_injection(self):
        self.assertEqual(rdk_x5_remote._safe_linux_path("~/yolo-team/rdk", "目录"), "~/yolo-team/rdk")
        self.assertEqual(rdk_x5_remote._safe_linux_path("/home/sunrise/rdk", "目录"), "/home/sunrise/rdk")
        for unsafe in ("~/jobs;reboot", "~/../root", "~/jobs with spaces", "/"):
            with self.assertRaises(ValueError):
                rdk_x5_remote._safe_linux_path(unsafe, "目录")
        with self.assertRaises(ValueError):
            rdk_x5_remote._safe_host("192.168.1.8;reboot")

    @patch("rdk_x5_remote._require_program", return_value="ssh")
    def test_board_probe_uses_batch_ssh_and_runtime_check(self, _program):
        command = rdk_x5_remote.build_board_probe_command("192.168.1.88", "sunrise", "22")
        self.assertIn("BatchMode=yes", command)
        self.assertIn("StrictHostKeyChecking=accept-new", command)
        self.assertIn("sunrise@192.168.1.88", command)
        self.assertIn("hbm_runtime", command[-1])
        self.assertIn("RDK_BOARD_READY=1", command[-1])

    def test_manifest_records_compile_and_board_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "export" / "rdk-x5-npu-bundle"
            bundle.mkdir(parents=True)
            model = bundle / "output" / "model_bayese_640x640_nv12.bin"
            model.parent.mkdir()
            model.write_bytes(b"bin")
            manifest = bundle.parent / "model.manifest.json"
            manifest.write_text(json.dumps({"artifact": str(bundle), "vendor_conversion": {"status": "conversion_required"}}), encoding="utf-8")

            updated = rdk_x5_remote._update_manifest(bundle, model, {"status": "passed", "host": "rdk-x5"})

            self.assertEqual(updated, manifest)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["final_artifact"], str(model))
            self.assertEqual(data["vendor_conversion"]["status"], "board_validated")
            self.assertEqual(data["board_validation"]["host"], "rdk-x5")


if __name__ == "__main__":
    unittest.main()
