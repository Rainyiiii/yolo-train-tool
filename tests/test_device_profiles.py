import unittest

from device_profiles import DEVICE_PROFILES, get_device_profile, public_device_profiles, resolve_export_format


class DeviceProfilesTest(unittest.TestCase):
    def test_every_profile_has_a_valid_recommended_format(self):
        for profile_id, profile in DEVICE_PROFILES.items():
            with self.subTest(profile=profile_id):
                self.assertIn(profile["recommended_format"], profile["formats"])
                self.assertEqual(resolve_export_format(profile_id), profile["recommended_format"])

    def test_known_edge_targets_are_present(self):
        self.assertEqual(get_device_profile("raspberry_pi")["recommended_format"], "ncnn")
        self.assertEqual(get_device_profile("raspberry_pi_4")["recommended_input"], [416, 416])
        self.assertEqual(get_device_profile("raspberry_pi_5")["recommended_input"], [640, 640])
        self.assertEqual(get_device_profile("rockchip_rknn")["default_chip"], "rk3588")
        self.assertEqual(get_device_profile("drobotics_rdk")["default_chip"], "x5")
        self.assertEqual(get_device_profile("drobotics_rdk_x5")["final_artifact"], "Bayes-e INT8 .bin")
        self.assertTrue(get_device_profile("drobotics_rdk_x5")["vendor_ptq"])
        self.assertEqual(get_device_profile("maixcam")["opset"], 11)

    def test_legacy_profiles_remain_api_compatible_and_are_marked_hidden(self):
        profiles = {profile["id"]: profile for profile in public_device_profiles()}
        self.assertTrue(profiles["raspberry_pi"]["hidden"])
        self.assertTrue(profiles["drobotics_rdk"]["hidden"])
        self.assertFalse(profiles["raspberry_pi_4"].get("hidden", False))
        self.assertFalse(profiles["raspberry_pi_5"].get("hidden", False))
        self.assertFalse(profiles["drobotics_rdk_x5"].get("hidden", False))

    def test_unsupported_format_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_export_format("raspberry_pi", "rknn")


if __name__ == "__main__":
    unittest.main()
