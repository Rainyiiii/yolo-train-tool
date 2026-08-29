from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from annotation_server import AnnotationHTTPServer
from annotation_store import AnnotationStore
from train_panel import PanelHTTPServer, PanelHandler


class ServerSingleInstanceTest(unittest.TestCase):
    def test_training_panel_rejects_a_second_listener_on_the_same_port(self) -> None:
        first = PanelHTTPServer(("127.0.0.1", 0), PanelHandler)
        try:
            with self.assertRaises(OSError):
                PanelHTTPServer(("127.0.0.1", first.server_address[1]), PanelHandler)
        finally:
            first.server_close()

    def test_annotation_server_rejects_a_second_listener_on_the_same_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = AnnotationStore(Path(temporary) / "annotation")
            first = AnnotationHTTPServer(("127.0.0.1", 0), store, False)
            try:
                with self.assertRaises(OSError):
                    AnnotationHTTPServer(("127.0.0.1", first.server_address[1]), store, False)
            finally:
                first.server_close()


if __name__ == "__main__":
    unittest.main()
