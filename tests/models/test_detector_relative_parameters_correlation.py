import unittest
from parameterized import parameterized
from models.dayabay_v0 import model_dayabay_v0


class CorrelatedTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model = model_dayabay_v0(close=True, strict=False)
        storage = model.storage
        cls.parameters_all = storage("parameters.all")
        cls.parameters_normalized = storage("parameters.normalized")

        cls.efficiency_factor_path = "detector.detector_relative.{detector}.efficiency_factor"
        cls.energy_scale_path = "detector.detector_relative.{detector}.energy_scale_factor"

    def set_up(self, detector0, detector1):
        efficiency_factor_path = self.efficiency_factor_path.format(detector=detector0)
        energy_scale_path = self.energy_scale_path.format(detector=detector1)

        self.efficiency_factor = self.parameters_all[efficiency_factor_path]
        self.energy_scale_factor = self.parameters_all[energy_scale_path]

        self.efficiency_factor_normalized = self.parameters_normalized[efficiency_factor_path]
        self.energy_scale_factor_normalized = self.parameters_normalized[energy_scale_path]

        self.efficiency_factor_init_value = self.efficiency_factor.value
        self.energy_scale_factor_init_value = self.energy_scale_factor.value

    def reset_normalized_values(self):
        self.efficiency_factor_normalized.value = 0
        self.energy_scale_factor_normalized.value = 0

    @parameterized.expand([("AD11", "AD11")])
    def test_correlated(self, detector0, detector1):
        self.set_up(detector0, detector1)

        self.reset_normalized_values()
        for value in [-1, 1]:
            self.efficiency_factor_normalized.value = value
            self.assertNotEqual(
                self.efficiency_factor_init_value,
                self.efficiency_factor.value,
                "Efficiency factor has not changed after changing",
            )
            self.assertNotEqual(
                self.energy_scale_factor_init_value,
                self.energy_scale_factor.value,
                "Energy scale factor has not changed after changing efficiency factor",
            )

        self.reset_normalized_values()
        for value in [-1, 1]:
            self.energy_scale_factor_normalized.value = value
            self.assertNotEqual(
                self.energy_scale_factor_init_value,
                self.energy_scale_factor.value,
                "Energy scale factor has not changed after changing",
            )
            self.assertEqual(
                self.efficiency_factor_init_value,
                self.efficiency_factor.value,
                "Efficiency factor has changed after changing energy scale factor",
            )

    @parameterized.expand([("AD11", "AD12")])
    def test_uncorrelated(self, detector0, detector1):
        self.set_up(detector0, detector1)

        self.reset_normalized_values()
        for value in [-1, 1]:
            self.efficiency_factor_normalized.value = value
            self.assertNotEqual(
                self.efficiency_factor_init_value,
                self.efficiency_factor.value,
                "Efficiency factor has not changed after changing",
            )
            self.assertEqual(
                self.energy_scale_factor_init_value,
                self.energy_scale_factor.value,
                "Energy scale factor changed after changing unrelated efficiency factor",
            )

        self.reset_normalized_values()
        for value in [-1, 1]:
            self.energy_scale_factor_normalized.value = value
            self.assertNotEqual(
                self.energy_scale_factor_init_value,
                self.energy_scale_factor.value,
                "Energy scale factor has not changed after changing",
            )
            self.assertEqual(
                self.efficiency_factor_init_value,
                self.efficiency_factor.value,
                "Efficiency factor changed after changing energy scale factor of unrelated detector",
            )


if __name__ == "__main__":
    unittest.main()
