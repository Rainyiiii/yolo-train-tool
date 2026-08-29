import unittest

from train_panel import DEFAULT_VALUES, HTML_PAGE, build_export_cmd, build_rdk_remote_cmd


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

    def test_rdk_commands_form_a_continuous_wsl_and_board_flow(self):
        values = {
            **DEFAULT_VALUES,
            "rdk_bundle_dir": "D:/exports/rdk-x5-npu-bundle",
            "rdk_bin_path": "D:/exports/rdk-x5-npu-bundle/output/model.bin",
            "rdk_board_host": "192.168.1.88",
        }
        compile_command = [str(item) for item in build_rdk_remote_cmd("rdk_wsl_compile", values)]
        deploy_command = [str(item) for item in build_rdk_remote_cmd("rdk_board_deploy", values)]
        self.assertIn("wsl-compile", compile_command)
        self.assertIn("Ubuntu-22.04", compile_command)
        self.assertIn("board-deploy", deploy_command)
        self.assertIn("sunrise", deploy_command)
        self.assertNotIn("--identity-file", deploy_command)

    def test_rdk_remote_controls_are_visible_in_the_deployment_page(self):
        for fragment in (
            'id="rdk_wsl_distro"',
            'id="rdk_bundle_dir"',
            'id="rdk_bin_path"',
            'id="rdk_board_host"',
            "runAction('rdk_wsl_compile')",
            "runAction('rdk_board_deploy')",
        ):
            self.assertIn(fragment, HTML_PAGE)


if __name__ == "__main__":
    unittest.main()
