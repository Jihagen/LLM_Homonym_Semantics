import math
import unittest
from pathlib import Path

import numpy as np

from analysis.context_revelation_trajectory import (
    STAGES,
    _pair_heldout_scores,
    build_prefix_stages,
)


class ContextTrajectoryTests(unittest.TestCase):
    def test_shared_stage_is_identical_within_each_sense_pair(self):
        records = build_prefix_stages(Path("data/paired_sentences.json"), "bank")
        self.assertEqual(len(records), 5 * 2 * len(STAGES))
        shared = [record for record in records if record["stage"] == 0]
        for pair_id in range(5):
            texts = [record["text"] for record in shared if record["pair_id"] == pair_id]
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], texts[1])

    def test_coincident_centroids_are_ties_not_false_errors(self):
        # Each sense contains the same two carrier representations.
        X = np.array([[0.0], [0.0], [1.0], [1.0]])
        labels = np.array([0, 1, 0, 1])
        pair_ids = np.array([0, 0, 1, 1])
        result = _pair_heldout_scores(X, labels, pair_ids)
        self.assertTrue(math.isnan(result["mean_margin_norm"]))
        self.assertEqual(result["tie_aware_accuracy"], 0.5)
        self.assertEqual(result["mean_centroid_distance"], 0.0)

    def test_separated_pairs_are_classified(self):
        X = np.array([[-2.0], [2.0], [-1.0], [1.0], [-3.0], [3.0]])
        labels = np.array([0, 1, 0, 1, 0, 1])
        pair_ids = np.array([0, 0, 1, 1, 2, 2])
        result = _pair_heldout_scores(X, labels, pair_ids)
        self.assertEqual(result["tie_aware_accuracy"], 1.0)
        self.assertGreater(result["mean_margin_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
