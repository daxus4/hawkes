from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tick.hawkes as hk

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class HawkesEventDatabaseExtractor(EventDatabaseExtractor):
    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        end_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
        prediction_period_duration: pd.Timedelta,
        simulation_period_duration: pd.Timedelta,
        warm_up_period_duration: pd.Timedelta,
        best_decay: Optional[float] = None,
    ) -> None:
        super().__init__(
            orderbook_df,
            start_time_simulation,
            end_time_simulation,
            coe_training_duration,
            training_duration,
            simulation_duration
        )

        self._prediction_period_duration = prediction_period_duration
        self._prediction_period_duration_seconds = prediction_period_duration.total_seconds()
        self._simulation_period_duration_seconds = int(simulation_period_duration.total_seconds())
        self._warm_up_period_duration_seconds = warm_up_period_duration.total_seconds()

        self._best_decay = best_decay

    def get_attack_times(
        self, 
    ) -> np.ndarray:
        attack_times = pd.to_datetime(
            self._orderbook_df['Timestamp'], unit='ms'
        ).to_numpy()

        end_time_simulation_with_offset = self._end_time_simulation + self._prediction_period_duration

        first_event_not_in_simulation = attack_times[attack_times > end_time_simulation_with_offset][0]
        attack_times = attack_times[
            (attack_times >= self._start_time_training) & (attack_times <= end_time_simulation_with_offset)
        ].copy()
        attack_times = np.append(attack_times, first_event_not_in_simulation)

        attack_times = np.array([dt.astype('datetime64[ms]').astype(float) / 1000 for dt in attack_times])

        return attack_times

    def get_training_attack_times(
        self,
        attack_times: np.ndarray,
    ) -> np.ndarray:
        return attack_times[
            attack_times <= self._start_time_simulation_timestamp
        ] - self._start_time_training_timestamp

    def get_best_decays_with_cross_validation(
        self, attack_times: np.ndarray, min_beta: float = 0.01, max_beta: float = 100.0
    ) -> float:
        beta_values = np.linspace(min_beta, max_beta, num=3000)

        best_beta = None
        best_score = float('-inf')

        #  cross validation per trovare il miglior beta
        for beta in beta_values:
            decays = np.array([[beta]])
            hawkes_model = hk.HawkesExpKern(decays=decays)
            hawkes_model.fit([attack_times])
            score = hawkes_model.score([attack_times])
            if score > best_score:
                best_score = score
                best_beta = beta
        
        return best_beta

    def get_hawkes_parameters_trained(
        self,
        attack_times: np.ndarray,
        decays: float = 0.06162109, 
    ) -> Tuple[float, List[np.ndarray], np.ndarray]:
        hawkes = hk.HawkesExpKern(decays=decays)
        hawkes.fit([attack_times])

        baseline = hawkes.baseline
        adjacency = hawkes.adjacency
        abb=baseline[0]
        baa= [np.array([decays])]
        caa=np.array([[adjacency[0,0]]])

        return abb, baa, caa

    def get_hawkes_results(
        self,
        attack_times: np.ndarray,
        abb: float, baa: List[np.ndarray], caa:np.ndarray,
    ) -> pd.DataFrame:
        simulation_results_map = {
            'real_next_event_timestamp': [],
            'predicted_next_event_timestamp': [],
        }

        for i in range(self._simulation_period_duration_seconds):
            current_start_time_simulation_timestamp = self._start_time_simulation_timestamp + i

            start_warm_up_period = current_start_time_simulation_timestamp - self._warm_up_period_duration_seconds
            end_warm_up_period = current_start_time_simulation_timestamp

            current_attack_times = attack_times[
                (attack_times >= start_warm_up_period) & (attack_times < end_warm_up_period)
            ].copy()
            current_attack_times = current_attack_times - start_warm_up_period

            sim_hawkes = self.get_hawkes_simulation(
                current_attack_times, self._warm_up_period_duration_seconds, self._prediction_period_duration_seconds, abb, baa, caa
            )

            predicted_timestamps = self.get_predicted_timestamps(sim_hawkes, self._warm_up_period_duration_seconds)
            predicted_next_event_timestamp = (
                predicted_timestamps[0]
                if len(predicted_timestamps) > 0
                else (end_warm_up_period - start_warm_up_period + self._prediction_period_duration_seconds)
            )

            real_next_event_timestamp = self.get_nearest_value(
                attack_times[attack_times > end_warm_up_period], predicted_next_event_timestamp + start_warm_up_period
            ) - start_warm_up_period

            simulation_results_map['real_next_event_timestamp'].append(start_warm_up_period + real_next_event_timestamp)
            simulation_results_map['predicted_next_event_timestamp'].append(start_warm_up_period + predicted_next_event_timestamp)

        return pd.DataFrame(simulation_results_map)

    def get_hawkes_simulation(
        self,
        attack_times: np.ndarray,
        warm_up_period_duration: float,
        abb: float,
        baa: List[np.ndarray],
        caa:np.ndarray,
        seed: int = 1039
    ) -> hk.SimuHawkesExpKernels:
        sim_hawkes = hk.SimuHawkesExpKernels(
            adjacency=caa, decays=baa, baseline=[abb], end_time=warm_up_period_duration, seed=seed
        )
        sim_hawkes.track_intensity(1)

        sim_hawkes.set_timestamps([attack_times], warm_up_period_duration)
        t_max = warm_up_period_duration + self._prediction_period_duration_seconds
        sim_hawkes.end_time = t_max

        sim_hawkes.simulate()

        return sim_hawkes

    def get_predicted_timestamps(
        self,
        sim_hawkes: hk.SimuHawkesExpKernels,
    ) -> np.ndarray:
        ay = sim_hawkes.timestamps
        ay_array = np.array(ay)
        condition = ay_array > self._warm_up_period_duration_seconds
        filtered_list = ay_array[condition]

        return filtered_list

    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        attack_times = self.get_attack_times()

        training_attack_times = self.get_training_attack_times(
            attack_times
        )
        if self._best_decay is None:
            self._best_decay = self.get_best_decays_with_cross_validation(training_attack_times)
            
        abb, baa, caa = self.get_hawkes_parameters_trained(training_attack_times, self._best_decay)

        hawkes_result_df = self.get_hawkes_results(
            attack_times, abb, baa, caa,
        )

        return hawkes_result_df

    @property
    def best_decay(self) -> float:
        return self._best_decay
