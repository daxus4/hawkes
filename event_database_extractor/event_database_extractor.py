import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd


class EventDatabaseExtractor(ABC):
    @classmethod
    def get_nearest_value(cls, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        end_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
    ) -> None:
        self._orderbook_df = orderbook_df
        self._start_time_simulation = start_time_simulation
        self._start_time_simulation_str = self._start_time_simulation.strftime('%H%M%S')

        self._start_time_simulation_timestamp = self._start_time_simulation.timestamp()

        self._start_time_training = self._start_time_simulation - training_duration
        self._start_time_training_timestamp = self._start_time_training.timestamp()

        self._end_time_simulation = self._start_time_simulation + simulation_duration
        self._end_time_simulation_timestamp = end_time_simulation.timestamp()

        self._coe_training_start_time = self._start_time_simulation - coe_training_duration
        self._coe_training_start_time_timestamp = self._coe_training_start_time.timestamp()

        self._coe_test_df = pd.DataFrame()

    def _compute_coe_training_df(self) -> None:
        coe_training_start_time_timestamp_milliseconds = self._coe_training_start_time_timestamp * 1000
        coe_training_end_time_timestamp_milliseconds = self._start_time_simulation_timestamp * 1000
        df = self._orderbook_df[
            (df['Timestamp'] >= coe_training_start_time_timestamp_milliseconds) & 
            (df['Timestamp'] < coe_training_end_time_timestamp_milliseconds)
        ].copy()
        df = df[['Timestamp', 'BaseImbalance', 'Return']].copy()
        df['Timestamp'] = (df['Timestamp'] - coe_training_start_time_timestamp_milliseconds) / 1000
        df = df.dropna()

        self._coe_training_df = df

    @abstractmethod
    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        pass

    def _compute_coe_test_df(self) -> None:
        coe_timestamp_test_df = self._get_coe_timestamp_test_df()
        self._coe_test_df = coe_timestamp_test_df.merge(
            self._orderbook_df,
            how='left',
            left_on='RealTimestampNotScaled',
            right_on='Timestamp',
        )
        self._coe_test_df = self._coe_test_df[
            ['Timestamp_x', 'BaseImbalance', 'Return', 'RealTimestampNotScaled']
        ]

        self._coe_test_df['RealTimestampNotScaled'] = (
            self._coe_test_df['RealTimestampNotScaled'] / 1000
        ) - self._coe_training_start_time_timestamp
        self._coe_test_df.rename(
            columns={
                'Timestamp_x': 'Timestamp', "RealTimestampNotScaled": 'RealTimestamp'
            },
            inplace=True
        )

    def get_index_positions_without_old_greater_predictions(
        self,
        series: pd.Series
    ) -> pd.Series:
        to_delete_index_positions = []
        for i in range(0, len(series)):
            for j in range(i+1, len(series)):
                if series.iloc[i] > series.iloc[j]:
                    to_delete_index_positions.append(i)
                    break
                
        return to_delete_index_positions

    def get_dataframe_without_index_positions(
        df: pd.DataFrame, index_positions: List
    ) -> pd.DataFrame:
        return df.drop(df.index[index_positions])

    def get_complete_coe_df(self) -> pd.DataFrame:
        self._compute_coe_training_df()
        self._compute_coe_test_df()

        testing_coe_index_positions = self.get_index_positions_without_old_greater_predictions(
            self._coe_test_df['Timestamp']
        )
        self._coe_test_df = self.get_dataframe_without_index_positions(
            self._coe_test_df, testing_coe_index_positions
        )
        self._coe_test_df.sort_values(by='Timestamp', inplace=True)
        
        df = pd.concat([self._coe_training_df, self._coe_test_df])
        df['Timestamp'] = df['Timestamp'].round(3)
        df['RealTimestamp'] = df['RealTimestamp'].round(3)
        df = df.reset_index(drop=True)
        df.columns = ['1', '2', '3', '4']
        return df
