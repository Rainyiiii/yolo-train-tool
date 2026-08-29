from __future__ import annotations

import unittest
from unittest.mock import patch

from importlib.metadata import PackageNotFoundError

import system_check
from system_check import module_status


class SystemCheckTest(unittest.TestCase):
    def test_matching_distribution_version_is_ready(self) -> None:
        with (
            patch.object(system_check.importlib.util, "find_spec", return_value=object()),
            patch.object(system_check, "package_version", return_value="1.20.0"),
        ):
            status = module_status("onnxruntime", distribution="onnxruntime", requirement=">=1.18,<2")
        self.assertTrue(status["ok"])

    def test_incompatible_distribution_version_is_reported(self) -> None:
        with (
            patch.object(system_check.importlib.util, "find_spec", return_value=object()),
            patch.object(system_check, "package_version", return_value="5.0.0"),
        ):
            status = module_status("cv2", distribution="opencv-contrib-python", requirement=">=4.10,<5")
        self.assertFalse(status["ok"])
        self.assertEqual(status["reason"], "incompatible")

    def test_wrong_opencv_package_is_not_accepted_as_contrib(self) -> None:
        with (
            patch.object(system_check.importlib.util, "find_spec", return_value=object()),
            patch.object(system_check, "package_version", side_effect=PackageNotFoundError("opencv-contrib-python")),
        ):
            status = module_status("cv2", distribution="opencv-contrib-python", requirement=">=4.10,<5")
        self.assertFalse(status["ok"])
        self.assertEqual(status["reason"], "wrong-package")


if __name__ == "__main__":
    unittest.main()
