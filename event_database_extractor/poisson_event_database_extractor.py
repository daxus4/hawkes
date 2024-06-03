from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tick.hawkes as hk

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class PoissonEventDatabaseExtractor(EventDatabaseExtractor):
    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
        point_process_training_duration: pd.Timedelta,
        prediction_period_duration: pd.Timedelta,
        warm_up_period_duration: pd.Timedelta,
    ) -> None:
        super().__init__(
            orderbook_df,
            start_time_simulation,
            coe_training_duration,
            simulation_duration
        )

        self._start_time_training = self._start_time_simulation - point_process_training_duration
        self._start_time_training_timestamp = self._start_time_training.timestamp()
        self._prediction_period_duration = prediction_period_duration
        self._prediction_period_duration_seconds = prediction_period_duration.total_seconds()
        self._simulation_period_duration_seconds = int(self._simulation_duration.total_seconds())
        self._warm_up_period_duration_seconds = warm_up_period_duration.total_seconds()

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

    def get_hawkes_parameters_trained(
        self,
        attack_times: np.ndarray,
    ) -> float:
        return (attack_times.max() - attack_times.min()) / len(attack_times)
    
    def get_hawkes_results(
        self,
        attack_times: np.ndarray,
        intensity: float
    ) -> pd.DataFrame:
        simulation_results_map = {
            'Timestamp': [],
            'NearestEventTimestampNotScaled': [],
            'LastEventTimestampNotScaled': [],
            'RealNextEventTimestampNotScaled': [],
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
                current_attack_times, self._warm_up_period_duration_seconds, intensity
            )

            predicted_timestamps = self.get_predicted_timestamps(sim_hawkes)
            predicted_next_event_timestamp = (
                predicted_timestamps[0]
                if len(predicted_timestamps) > 0
                else (end_warm_up_period - start_warm_up_period + self._prediction_period_duration_seconds)
            )

            nearest_event_timestamp = self.get_nearest_value(
                attack_times, predicted_next_event_timestamp + start_warm_up_period
            ) - start_warm_up_period

            last_event_timestamp = self.get_last_value(
                attack_times, end_warm_up_period
            ) - start_warm_up_period

            real_next_event_timestamp = self.get_next_value(
                attack_times, end_warm_up_period
            ) - start_warm_up_period

            simulation_results_map['Timestamp'].append(start_warm_up_period + predicted_next_event_timestamp)
            simulation_results_map['NearestEventTimestampNotScaled'].append(start_warm_up_period + nearest_event_timestamp)
            simulation_results_map['LastEventTimestampNotScaled'].append(start_warm_up_period + last_event_timestamp)
            simulation_results_map['RealNextEventTimestampNotScaled'].append(start_warm_up_period + real_next_event_timestamp)


        df = pd.DataFrame(simulation_results_map)
        
        df['Timestamp'] = (df['Timestamp'] * 1000).astype(int)
        df['NearestEventTimestampNotScaled'] = (df['NearestEventTimestampNotScaled'] * 1000).astype(int)
        df['LastEventTimestampNotScaled'] = (df['LastEventTimestampNotScaled'] * 1000).astype(int)
        df['RealNextEventTimestampNotScaled'] = (df['RealNextEventTimestampNotScaled'] * 1000).astype(int)

        return df

    def get_hawkes_simulation(
        self,
        attack_times: np.ndarray,
        warm_up_period_duration: float,
        abb: float,
        seed: int = 1039,
    ) -> hk.SimuHawkesExpKernels:
        sim_hawkes = hk.SimuHawkesExpKernels(
            adjacency=np.array([[0]]), decays=[np.array([0])], baseline=[abb], end_time=warm_up_period_duration, seed=seed
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
            
        intensity = self.get_hawkes_parameters_trained(training_attack_times)

        hawkes_result_df = self.get_hawkes_results(
            attack_times, intensity
        )

        return hawkes_result_df
