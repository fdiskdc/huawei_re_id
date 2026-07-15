import unittest

import numpy as np

from attention_distribution import (
    NORMALIZATION,
    attention_distribution_cache_key,
    build_attention_distribution,
)


class AttentionDistributionTest(unittest.TestCase):
    def setUp(self):
        self.names = ["Am", "m6A"]
        self.probabilities = [0.6, 0.3]
        self.predictions = {0: True, 1: False}
        self.thresholds = {0: 0.51, 1: 0.26}

    def build(self, sequence, weights, left_padding=0, left_trimming=0):
        return build_attention_distribution(
            original_sequence=sequence,
            attn_weights=np.asarray(weights),
            probs_12class=self.probabilities,
            predictions_12class=self.predictions,
            thresholds_12class=self.thresholds,
            class_names=self.names,
            left_padding=left_padding,
            left_trimming=left_trimming,
        )

    def test_short_sequence_removes_padding_and_renormalizes(self):
        result = self.build(
            "ACG",
            [
                [50, 1, 2, 3, 50],
                [50, 3, 3, 6, 50],
            ],
            left_padding=1,
        )

        self.assertEqual(result["sequence_length"], 3)
        self.assertEqual(result["modeled_range"], {"start": 0, "end": 3})
        self.assertEqual(result["normalization"], NORMALIZATION)
        self.assertEqual(result["classes"][0]["attention"], [0.16666667, 0.33333333, 0.5])
        self.assertAlmostEqual(sum(result["classes"][1]["attention"]), 1.0)

    def test_exact_model_length_keeps_all_positions(self):
        result = self.build("ACGU", [[1, 1, 1, 1], [4, 3, 2, 1]])

        self.assertEqual(result["modeled_range"], {"start": 0, "end": 4})
        self.assertEqual(len(result["classes"][0]["attention"]), 4)
        self.assertTrue(result["classes"][0]["is_predicted"])
        self.assertFalse(result["classes"][1]["is_predicted"])
        self.assertEqual(result["classes"][0]["probability"], 0.6)
        self.assertEqual(result["classes"][1]["threshold"], 0.26)

    def test_long_sequence_reports_original_coordinate_range(self):
        result = self.build(
            "ACGUAC",
            [[1, 2, 3, 4], [4, 3, 2, 1]],
            left_trimming=1,
        )

        self.assertEqual(result["sequence_length"], 6)
        self.assertEqual(result["modeled_range"], {"start": 1, "end": 5})
        self.assertEqual(len(result["classes"][0]["attention"]), 4)

    def test_cache_key(self):
        self.assertEqual(
            attention_distribution_cache_key("abc"),
            "attention_distribution:abc",
        )


if __name__ == "__main__":
    unittest.main()
