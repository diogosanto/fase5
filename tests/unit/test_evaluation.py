import unittest

import pandas as pd

from src.training.evaluation import regression_metrics, segment_metrics


class EvaluationTests(unittest.TestCase):
    def test_regression_metrics_include_tail_error_and_bias(self) -> None:
        metrics = regression_metrics([100, 200, 300], [110, 190, 330])

        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["mae"], 50 / 3)
        self.assertIn("p95_absolute_error", metrics)
        self.assertIn("bias", metrics)

    def test_segment_metrics_group_by_period_and_target_band(self) -> None:
        frame = pd.DataFrame(
            {
                "bairro": ["A", "A", "B", "B"],
                "ano": [2024, 2024, 2025, 2025],
                "mes": [1, 1, 2, 2],
            }
        )

        metrics = segment_metrics(frame, [100, 200, 300, 400], [100, 210, 330, 380], min_group_size=1)

        self.assertIn("by_period", metrics)
        self.assertIn("by_target_band", metrics)
        self.assertIn("worst_bairros_by_mae", metrics)
        self.assertTrue(metrics["by_period"])


if __name__ == "__main__":
    unittest.main()
