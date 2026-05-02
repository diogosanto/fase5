import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from src.training import promote_model


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class PromoteModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work_dir = WORKSPACE_ROOT / ".test_artifacts" / "promote_model"
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_blocks_when_mae_exceeds_max_mae(self) -> None:
        candidate = self._model_dir("candidate", mae=60_000)

        with self.assertRaisesRegex(ValueError, "acima do limite"):
            promote_model.assert_promotion_criteria(
                candidate_path=candidate,
                to_env="test",
                improvement_pct=5.0,
                max_mae=50_000,
            )

    def test_approves_without_active_model_when_under_max_mae(self) -> None:
        candidate = self._model_dir("candidate", mae=40_000)

        with patch.object(promote_model, "get_active_model", return_value=None):
            promote_model.assert_promotion_criteria(
                candidate_path=candidate,
                to_env="test",
                improvement_pct=5.0,
                max_mae=50_000,
            )

    def test_blocks_when_candidate_does_not_improve_active_model(self) -> None:
        candidate = self._model_dir("candidate", mae=97_000)
        active = self._model_dir("active", mae=100_000)

        with patch.object(promote_model, "get_active_model", return_value=active):
            with self.assertRaisesRegex(ValueError, "necessario <="):
                promote_model.assert_promotion_criteria(
                    candidate_path=candidate,
                    to_env="prod",
                    improvement_pct=5.0,
                    max_mae=None,
                )

    def test_approves_when_candidate_improves_active_model(self) -> None:
        candidate = self._model_dir("candidate", mae=94_000)
        active = self._model_dir("active", mae=100_000)

        with patch.object(promote_model, "get_active_model", return_value=active):
            promote_model.assert_promotion_criteria(
                candidate_path=candidate,
                to_env="prod",
                improvement_pct=5.0,
                max_mae=None,
            )

    def test_skips_active_model_metric_when_improvement_pct_is_zero(self) -> None:
        candidate = self._model_dir("candidate", mae=94_000)
        active = self.work_dir / "active_without_metrics"
        active.mkdir(parents=True, exist_ok=True)

        with patch.object(promote_model, "get_active_model", return_value=active):
            promote_model.assert_promotion_criteria(
                candidate_path=candidate,
                to_env="test",
                improvement_pct=0.0,
                max_mae=None,
            )

    def _model_dir(self, name: str, mae: float) -> Path:
        model_dir = self.work_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump({"mae": mae}, file)
        return model_dir


if __name__ == "__main__":
    unittest.main()
