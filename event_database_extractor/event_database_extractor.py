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

    @classmethod
    def get_last_value(cls, array: np.ndarray, limit_value: float) -> float:
        return array[array <= limit_value][-1]
    
    @classmethod
    def get_next_value(cls, array: np.ndarray, this_value: float) -> float:
        return array[array > this_value][0]

    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
    ) -> None:
        self._orderbook_df = orderbook_df

        self._start_time_simulation = start_time_simulation
        self._start_time_simulation_timestamp = self._start_time_simulation.timestamp()

        self._simulation_duration = simulation_duration

        self._end_time_simulation = self._start_time_simulation + simulation_duration
        self._end_time_simulation_timestamp = self._end_time_simulation.timestamp()

        self._coe_training_start_time = (
            self._start_time_simulation - coe_training_duration
        )
        self._coe_training_start_time_timestamp = (
            self._coe_training_start_time.timestamp()
        )

        self._coe_test_df = pd.DataFrame()

    def _compute_coe_training_df(self) -> None:
        coe_training_start_time_timestamp_milliseconds = (
            self._coe_training_start_time_timestamp * 1000
        )
        coe_training_end_time_timestamp_milliseconds = (
            self._start_time_simulation_timestamp * 1000
        )
        df = self._orderbook_df[
            (
                self._orderbook_df["Timestamp"]
                >= coe_training_start_time_timestamp_milliseconds
            )
            & (
                self._orderbook_df["Timestamp"]
                < coe_training_end_time_timestamp_milliseconds
            )
        ].copy()
        df = df[["Timestamp", "BaseImbalance", "Return"]].copy()
        df["Timestamp"] = (
            df["Timestamp"] - coe_training_start_time_timestamp_milliseconds
        ) / 1000
        df = df.dropna()

        self._coe_training_df = df

    @abstractmethod
    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        pass

    def _compute_coe_test_df(
        self, use_near_event_timestamp_for_returns: bool = True
    ) -> None:
        coe_timestamp_test_df = self._get_coe_timestamp_test_df()
        coe_timestamp_test_df["BaseImbalance"] = 0
        coe_timestamp_test_df["Return"] = 0

        near_event_feature = (
            "Return" if use_near_event_timestamp_for_returns else "BaseImbalance"
        )
        last_event_feature = (
            "BaseImbalance" if use_near_event_timestamp_for_returns else "Return"
        )

        for i in range(0, len(coe_timestamp_test_df)):
            coe_timestamp_test_df[near_event_feature].iloc[i] = self._orderbook_df[
                self._orderbook_df["Timestamp"]
                == coe_timestamp_test_df["NearestEventTimestampNotScaled"].iloc[i]
            ][near_event_feature].values[0]
            coe_timestamp_test_df[last_event_feature].iloc[i] = self._orderbook_df[
                self._orderbook_df["Timestamp"]
                == coe_timestamp_test_df["LastEventTimestampNotScaled"].iloc[i]
            ][last_event_feature].values[0]

        self._coe_test_df = coe_timestamp_test_df[
            [
                "Timestamp",
                "BaseImbalance",
                "Return",
                "LastEventTimestampNotScaled",
                "NearestEventTimestampNotScaled",
                "RealNextEventTimestampNotScaled"
            ]
        ]

        self._coe_test_df["Timestamp"] = (
            self._coe_test_df["Timestamp"] / 1000
        ) - self._coe_training_start_time_timestamp
        self._coe_test_df["LastEventTimestampNotScaled"] = (
            self._coe_test_df["LastEventTimestampNotScaled"] / 1000
        ) - self._coe_training_start_time_timestamp
        self._coe_test_df["NearestEventTimestampNotScaled"] = (
            self._coe_test_df["NearestEventTimestampNotScaled"] / 1000
        ) - self._coe_training_start_time_timestamp
        self._coe_test_df["RealNextEventTimestampNotScaled"] = (
            self._coe_test_df["RealNextEventTimestampNotScaled"] / 1000
        ) - self._coe_training_start_time_timestamp

    def get_complete_coe_df(
        self, use_near_event_timestamp_for_returns: bool = True
    ) -> pd.DataFrame:
        self._compute_coe_training_df()
        self._compute_coe_test_df(use_near_event_timestamp_for_returns)

        df = pd.concat([self._coe_training_df, self._coe_test_df])
        df["Timestamp"] = df["Timestamp"].round(3)
        df["LastEventTimestampNotScaled"] = df["LastEventTimestampNotScaled"].round(3)
        df["NearestEventTimestampNotScaled"] = df[
            "NearestEventTimestampNotScaled"
        ].round(3)
        df["RealNextEventTimestampNotScaled"] = df[
            "RealNextEventTimestampNotScaled"
        ].round(3)
        df = df.reset_index(drop=True)
        return df
