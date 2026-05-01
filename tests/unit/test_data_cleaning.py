import importlib.util
from pathlib import Path
import unittest

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "2clean_all.py"
SPEC = importlib.util.spec_from_file_location("clean_all", MODULE_PATH)
cleaning = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cleaning)


class DataCleaningTests(unittest.TestCase):
    def test_repairs_common_mojibake(self) -> None:
        self.assertEqual(cleaning.repair_mojibake("N\u00c3\u00bamero"), "N\u00famero")

    def test_drops_embedded_header_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"n_do_cadastro": "N\u00b0 do Cadastro (SQL)", "data_de_transacao": "DATA DE TRANSA\u00c7\u00c3O"},
                {"n_do_cadastro": "123", "data_de_transacao": "2024-01-10"},
            ]
        )

        cleaned, dropped = cleaning.drop_embedded_header_rows(frame)

        self.assertEqual(dropped, 1)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["n_do_cadastro"], "123")

    def test_convert_date_prefers_brazilian_day_first_format(self) -> None:
        dates = cleaning.convert_date(pd.Series(["03/04/2024"]))

        self.assertEqual(dates.iloc[0].strftime("%Y-%m-%d"), "2024-04-03")

    def test_profile_records_duplicate_rows_removed(self) -> None:
        frame = pd.DataFrame(
            {
                "data_de_transacao": pd.to_datetime(["2024-01-01"]),
                "bairro": ["A"],
                "cep": ["12345"],
                "descricao_do_uso_iptu": ["TERRENO"],
                "area_do_terreno_m2": [100.0],
                "valor_venal_de_referencia": [100000.0],
            }
        )

        profile = cleaning.build_profile(
            frame,
            dropped_header_rows=1,
            dropped_duplicate_rows=2,
            dropped_non_positive_target_rows=3,
        )

        self.assertEqual(profile["quality_checks"]["embedded_header_rows_removed"], 1)
        self.assertEqual(profile["quality_checks"]["duplicate_rows_removed"], 2)
        self.assertEqual(profile["quality_checks"]["non_positive_target_rows_removed"], 3)

    def test_drops_non_positive_target_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "valor_venal_de_referencia": [100000.0, 0.0, -10.0],
                "bairro": ["A", "B", "C"],
            }
        )

        cleaned, dropped = cleaning.drop_non_positive_target_rows(frame)

        self.assertEqual(dropped, 2)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["bairro"], "A")


if __name__ == "__main__":
    unittest.main()
