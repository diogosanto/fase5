import unittest

import pandas as pd

from src.training.splits import iter_temporal_backtest_splits, split_temporal_holdout


class TemporalHoldoutTests(unittest.TestCase):
    def test_temporal_holdout_uses_latest_rows_for_test(self) -> None:
        frame = pd.DataFrame(
            {
                "ano": [2023, 2023, 2024, 2024, 2025],
                "mes": [1, 2, 1, 2, 1],
                "bairro": ["A", "A", "A", "A", "A"],
                "cep_prefixo": ["1", "1", "1", "1", "1"],
                "area_do_terreno_m2": [10, 20, 30, 40, 50],
                "valor_venal_de_referencia": [100, 200, 300, 400, 500],
            }
        )

        train_df, test_df = split_temporal_holdout(frame)

        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(test_df), 1)
        self.assertEqual(int(test_df.iloc[0]["ano"]), 2025)
        self.assertEqual(int(test_df.iloc[0]["mes"]), 1)

    def test_backtest_splits_train_only_on_prior_periods(self) -> None:
        frame = pd.DataFrame(
            {
                "ano": [2023, 2023, 2024, 2024, 2025, 2025],
                "mes": [1, 2, 1, 2, 1, 2],
                "bairro": ["A"] * 6,
                "cep_prefixo": ["1"] * 6,
                "area_do_terreno_m2": [10, 20, 30, 40, 50, 60],
                "valor_venal_de_referencia": [100, 200, 300, 400, 500, 600],
            }
        )

        splits = iter_temporal_backtest_splits(frame, windows=2, min_train_periods=2)

        self.assertEqual([period for period, _, _ in splits], ["2025-01", "2025-02"])
        for period, train_df, test_df in splits:
            train_periods = train_df["ano"].astype(int) * 100 + train_df["mes"].astype(int)
            test_period = int(period.replace("-", ""))
            self.assertTrue((train_periods < test_period).all())
            self.assertTrue(((test_df["ano"].astype(int) * 100 + test_df["mes"].astype(int)) == test_period).all())


if __name__ == "__main__":
    unittest.main()
