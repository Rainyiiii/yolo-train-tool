import unittest

from device_profiles import DEVICE_PROFILES, get_device_profile, resolve_export_format


class DeviceProfilesTest(unittest.TestCase):
    def test_every_profile_has_a_valid_recommended_format(self):
        for profile_id, profile in DEVICE_PROFILES.items():
            with self.subTest(profile=profile_id):
                self.assertIn(profile["recommended_format"], profile["formats"])
                self.assertEqual(resolve_export_format(profile_id), profile["recommended_format"])

    def test_known_edge_targets_are_present(self):
        self.assertEqual(get_device_profile("raspberry_pi")["recommended_format"], "ncnn")
        self.assertEqual(get_device_profile("rockchip_rknn")["default_chip"], "rk3588")
        self.assertEqual(get_device_profile("drobotics_rdk")["default_chip"], "x5")
        self.assertEqual(get_device_profile("maixcam")["opset"], 11)

    def test_unsupported_format_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_export_format("raspberry_pi", "rknn")


if __name__ == "__main__":
    unittest.main()
