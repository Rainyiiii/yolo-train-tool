import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_panel import LabelTrackObject, label_tracking_quality, save_label_session_sample


class LabelTrackingSafetyTest(unittest.TestCase):
    def test_normal_motion_keeps_high_quality(self):
        quality, warning = label_tracking_quality((100, 100, 80, 60), (108, 104, 80, 60), 640, 480)
        self.assertGreaterEqual(quality, 0.7)
        self.assertEqual(warning, "")

    def test_large_jump_is_flagged(self):
        quality, warning = label_tracking_quality((10, 10, 30, 30), (400, 300, 30, 30), 640, 480)
        self.assertLess(quality, 0.35)
        self.assertIn("位置突变", warning)

    def test_partial_target_loss_never_writes_incomplete_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = {
                "objects": [
                    LabelTrackObject(1, "person", (10, 10, 20, 20), object(), ok=True),
                    LabelTrackObject(2, "car", (40, 40, 20, 20), object(), ok=False),
                ],
                "frame": np.zeros((100, 100, 3), dtype=np.uint8),
                "source_image_path": None,
                "annotations_dir": root / "annotations",
                "images_dir": root / "images",
                "prefix": "track",
                "frame_index": 2,
                "jpeg_quality": 95,
                "saved": 0,
                "review_skipped": 0,
                "last_warning": "",
                "last_auto_saved": False,
            }
            session["annotations_dir"].mkdir()
            session["images_dir"].mkdir()
            self.assertFalse(save_label_session_sample(session, automatic=True))
            self.assertEqual(session["review_skipped"], 1)
            self.assertEqual(list(session["annotations_dir"].iterdir()), [])
            self.assertEqual(list(session["images_dir"].iterdir()), [])


if __name__ == "__main__":
    unittest.main()
