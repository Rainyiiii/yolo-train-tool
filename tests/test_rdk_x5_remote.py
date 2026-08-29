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
        script = rdk_x5_remote.wsl_probe_script(rdk_x5_remote.DEFAULT_VENV)
        self.assertIn('test "$ARCH" = "x86_64"', script)
        self.assertIn("RDK_WSL_READY=1", script)
        self.assertNotIn(script, command)

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
