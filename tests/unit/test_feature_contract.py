import unittest

import pandas as pd

from src.features.build_features_minimal import build_feature_contract, validate_feature_contract


class FeatureContractTests(unittest.TestCase):
    def test_feature_contract_rejects_forbidden_columns(self) -> None:
        frame = self._valid_frame()
        frame["valor_m2"] = 123.0

        contract = build_feature_contract(frame)

        self.assertFalse(contract["passed"])
        with self.assertRaisesRegex(ValueError, "colunas proibidas"):
            validate_feature_contract(contract)

    def test_feature_contract_accepts_valid_feature_dataset(self) -> None:
        contract = build_feature_contract(self._valid_frame())

        self.assertTrue(contract["passed"])
        self.assertEqual(contract["target_column"], "valor_venal_de_referencia")
        validate_feature_contract(contract)

    def _valid_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "bairro": [f"BAIRRO_{index}" for index in range(25)] * 40,
                "cep_prefixo": ["12345"] * 1000,
                "area_do_terreno_m2": [100.0] * 1000,
                "valor_venal_de_referencia": [100000.0 + index for index in range(1000)],
                "ano": [2023, 2023, 2024, 2024, 2025, 2025, 2026, 2026] * 125,
                "mes": [1, 2, 1, 2, 1, 2, 1, 2] * 125,
            }
        )


if __name__ == "__main__":
    unittest.main()
