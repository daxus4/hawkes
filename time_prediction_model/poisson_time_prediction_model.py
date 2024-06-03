from typing import Any, Dict, Tuple

import numpy as np

from time_prediction_model.hawkes_time_prediction_model import HawkesTimePredictionModel


class PoissonTimePredictionModel(HawkesTimePredictionModel):
    def __init__(self) -> None:
        self._parameters: Dict[str, Any] = dict()

    def get_best_beta_with_cross_validation(
        self, attack_times: np.ndarray, min_beta: float = 0.01, max_beta: float = 100.0
    ) -> float:
        return 0.0

    def _get_alpha_and_mu(
        self,
        attack_times: np.ndarray,
        beta: float, 
    ) -> Tuple[float, float]:
        alpha = 0.0
        mu = (attack_times.max() - attack_times.min()) / len(attack_times)

        return alpha, mu