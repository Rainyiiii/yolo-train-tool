import unittest

from train_panel import DEFAULT_VALUES, build_export_cmd


class DeploymentWorkflowTest(unittest.TestCase):
    def test_vendor_calibration_is_only_passed_to_rdk_x5(self):
        values = {**DEFAULT_VALUES, "deployment_calibration_dir": "C:/calibration"}
        values["deployment_target"] = "raspberry_pi_4"
        pi_command = [str(item) for item in build_export_cmd(values)]
        self.assertNotIn("--calibration-images", pi_command)

        values["deployment_target"] = "drobotics_rdk_x5"
        rdk_command = [str(item) for item in build_export_cmd(values)]
        self.assertIn("--calibration-images", rdk_command)
        self.assertEqual(rdk_command[rdk_command.index("--calibration-images") + 1], "C:/calibration")

    def test_export_size_is_independent_from_training_size(self):
        values = {
            **DEFAULT_VALUES,
            "img_width": "320",
            "img_height": "320",
            "export_img_width": "640",
            "export_img_height": "416",
        }
        command = [str(item) for item in build_export_cmd(values)]
        self.assertEqual(command[command.index("--imgsz") + 1], "416,640")


if __name__ == "__main__":
    unittest.main()
