import numpy as np
import pandas as pd

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class MovingAverageEventDatabaseExtractor(EventDatabaseExtractor):
    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        end_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
        moving_average_window_size_seconds: int,
    ) -> None:
        super().__init__(
            orderbook_df,
            start_time_simulation,
            end_time_simulation,
            coe_training_duration,
            training_duration,
            simulation_duration
        )

        self._moving_average_window_size_seconds = moving_average_window_size_seconds

    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        start_time_for_moving_average = (self._start_time_simulation_timestamp - self._moving_average_window_size_seconds) * 1000
        testing_df = self._orderbook_df[
        (self._orderbook_df['Timestamp'] >= start_time_for_moving_average) &
        (self._orderbook_df['Timestamp'] <= (
            self._end_time_simulation_timestamp + self._moving_average_window_size_seconds
        ) * 1000)
        ].copy()
        testing_df = testing_df[['Timestamp', 'BaseImbalance', 'Return']]
        testing_df["DeltaT"] = testing_df["Timestamp"].shift(-1) - testing_df["Timestamp"]

        moving_average_window_size_milliseconds = self._moving_average_window_size_seconds * 1000

        df_map = {
            'Timestamp': list(),
            'RealTimestampNotScaled': list(),
        }

        for i in range(int(self._start_time_simulation_timestamp * 1000), int(self._end_time_simulation_timestamp * 1000), 1000):        
            moving_average = testing_df[
                (testing_df['Timestamp'] >= (i - moving_average_window_size_milliseconds)) &
                (testing_df['Timestamp'] <= i)
            ]['DeltaT'].mean()
            moving_average = int(moving_average) if not np.isnan(moving_average) else moving_average_window_size_milliseconds

            predicted_timestamp = i + moving_average
            df_map['Timestamp'].append(predicted_timestamp)
            nearest_timestamp = self.get_nearest_value(testing_df['Timestamp'].values, predicted_timestamp)
            df_map['RealTimestampNotScaled'].append(nearest_timestamp)

        moving_average_df = pd.DataFrame(df_map)
        moving_average_df['Timestamp'] = (moving_average_df['Timestamp']/1000 - self._coe_training_start_time_timestamp)
        return moving_average_df


