from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tick.hawkes as hk

from time_prediction_model.time_prediction_model import TimePredictionModel


class HawkesTimePredictionModel(TimePredictionModel):
    def __init__(self) -> None:
        self._parameters: Dict[str, Any] = dict()

    def get_best_beta_with_cross_validation(
        self, attack_times: np.ndarray, min_beta: float = 0.01, max_beta: float = 100.0
    ) -> float:
        beta_values = np.linspace(min_beta, max_beta, num=3000)

        best_beta = None
        best_score = float('-inf')

        for beta in beta_values:
            decays = np.array([[beta]])
            hawkes_model = hk.HawkesExpKern(decays=decays)
            hawkes_model.fit([attack_times])
            score = hawkes_model.score([attack_times])
            if score > best_score:
                best_score = score
                best_beta = beta
        
        return best_beta

    def _get_alpha_and_mu(
        self,
        attack_times: np.ndarray,
        beta: float, 
    ) -> Tuple[float, float]:
        hawkes = hk.HawkesExpKern(decays=beta)
        hawkes.fit([attack_times])

        baseline = hawkes.baseline
        adjacency = hawkes.adjacency
        mu = baseline[0]
        alpha = adjacency[0,0]

        return alpha, mu

    def train(
        self,
        times_in_seconds: np.ndarray,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        mu: Optional[float] = None,
    ) -> None:
        self._parameters['beta'] = (
            beta if beta is not None
            else self.get_best_beta_with_cross_validation(times_in_seconds)
        )

        if alpha is None or mu is None:
            (
                self._parameters['alpha'],
                self._parameters['mu']
            ) = self._get_alpha_and_mu(times_in_seconds, self._parameters['beta'])

        if alpha is not None:
            self._parameters['alpha'] = alpha
        if mu is not None:
            self._parameters['mu'] = mu

    def predict_next_event_time_from_current_time(
        self,
        time_in_seconds: np.ndarray,
        current_time_in_seconds: float,
        prediction_period_duration_seconds: float,
        seed: int = 1039,
    ) -> float:
        simulated_hawkes = self._get_hawkes_simulation(
            time_in_seconds, current_time_in_seconds + prediction_period_duration_seconds, seed
        )

        predicted_timestamps = self._get_predicted_timestamps(
            simulated_hawkes, current_time_in_seconds
        )

        return predicted_timestamps[0] if len(predicted_timestamps) > 0 else np.nan

    def _get_hawkes_simulation(
        self,
        attack_times: np.ndarray,
        max_simulated_time_seconds: float,
        seed: int,
    ) -> hk.SimuHawkesExpKernels:
        sim_hawkes = hk.SimuHawkesExpKernels(
            adjacency=self._get_alpha_converted_for_simulation(),
            decays=self._get_beta_converted_for_simulation(),
            baseline=self._get_mu_converted_for_simulation(),
            end_time=max_simulated_time_seconds,
            seed=seed
        )
        sim_hawkes.track_intensity(1)

        sim_hawkes.set_timestamps([attack_times], max_simulated_time_seconds)
        sim_hawkes.end_time = max_simulated_time_seconds

        sim_hawkes.simulate()

        return sim_hawkes

    def _get_predicted_timestamps(
        self,
        sim_hawkes: hk.SimuHawkesExpKernels,
        current_time_in_seconds: float,
    ) -> np.ndarray:
        ay = sim_hawkes.timestamps
        ay_array = np.array(ay)
        condition = ay_array > current_time_in_seconds
        filtered_list = ay_array[condition]

        return filtered_list

    def _get_alpha_converted_for_simulation(self) -> np.ndarray:
        return np.array([[self._parameters['alpha']]])
    
    def _get_beta_converted_for_simulation(self) -> List[np.ndarray]:
        return [np.array([self._parameters['beta']])]
    
    def _get_mu_converted_for_simulation(self) -> float:
        return self._parameters['mu']