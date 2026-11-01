from __future__ import annotations

import unittest

import numpy as np

from app.dsp import analyze


class SpectrogramTests(unittest.TestCase):
    def test_spectrogram_is_log_frequency_bounded_and_display_only(self):
        rate = 16_000
        time = np.arange(rate * 3, dtype=float) / rate
        samples = 0.4 * np.sin(2 * np.pi * 440 * time)
        report = analyze(samples, rate)
        spec = report["spectrogram"]
        self.assertEqual(len(spec["frequency_hz"]), 48)
        self.assertLessEqual(len(spec["values_u8"]), 600)
        self.assertEqual(len(spec["values_u8"][0]), 48)
        self.assertEqual(spec["frequency_scale"], "logarithmic")
        self.assertIn("not used by the classifier", spec["note"])
        frequencies = spec["frequency_hz"]
        self.assertTrue(all(left < right for left, right in zip(frequencies, frequencies[1:])))
        self.assertTrue(all(0 <= value <= 255 for row in spec["values_u8"] for value in row))

    def test_temporal_modulation_tracks_a_known_envelope_frequency(self):
        rate = 16_000
        time = np.arange(rate * 5, dtype=float) / rate
        envelope = 0.25 + 0.18 * np.sin(2 * np.pi * 2.2 * time)
        samples = envelope * np.sin(2 * np.pi * 300 * time)
        report = analyze(samples, rate)
        modulation = report["temporal_modulation"]
        self.assertAlmostEqual(modulation["peak_modulation_frequency_2_to_8_hz"], 2.2, delta=0.25)
        self.assertGreater(modulation["power_fraction_2_to_8_hz_of_0_5_to_20_hz"], 0.9)
        self.assertIn("not used", modulation["method"])


if __name__ == "__main__":
    unittest.main()
