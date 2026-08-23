from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import panel_service


class PanelServicePortTest(unittest.TestCase):
    def test_recorded_port_is_validated(self) -> None:
        self.assertEqual(panel_service.record_port({"port": 8991}), 8991)
        self.assertEqual(panel_service.record_port({"port": "9012"}), 9012)
        self.assertEqual(panel_service.record_port({"port": 70000}), 8989)
        self.assertEqual(panel_service.record_port({"port": "invalid"}), 8989)

    def test_running_service_reports_its_recorded_port(self) -> None:
        process = SimpleNamespace(pid=321)
        output = io.StringIO()
        with (
            patch.object(panel_service, "read_pid_record", return_value={"port": 8991}),
            patch.object(panel_service, "managed_process", return_value=process),
            patch.object(panel_service, "panel_ready", return_value=True) as ready,
            redirect_stdout(output),
        ):
            result = panel_service.start_panel(no_browser=True, port=8989)

        self.assertEqual(result, 0)
        ready.assert_called_once_with(8991)
        self.assertIn("http://127.0.0.1:8991/", output.getvalue())

    def test_running_panel_list_is_machine_readable(self) -> None:
        process = SimpleNamespace(pid=654, create_time=lambda: 123.5)
        output = io.StringIO()
        with (
            patch.object(
                panel_service,
                "discover_running_panels",
                return_value=[(process, 8991, "C:/old/train_panel.py")],
            ),
            redirect_stdout(output),
        ):
            result = panel_service.list_running_panels()

        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            [{
                "pid": 654,
                "create_time": 123.5,
                "port": 8991,
                "script": "C:/old/train_panel.py",
            }],
        )


if __name__ == "__main__":
    unittest.main()
