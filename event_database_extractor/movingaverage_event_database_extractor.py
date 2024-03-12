import numpy as np
import pandas as pd

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class MovingAverageEventDatabaseExtractor(EventDatabaseExtractor):
    def __init__(
        self,
        orderbook_df: pd.DataFrame,
        start_time_simulation: pd.Timestamp,
        coe_training_duration: pd.Timedelta,
        simulation_duration: pd.Timedelta,
        moving_average_window_size_seconds: int,
    ) -> None:
        super().__init__(
            orderbook_df,
            start_time_simulation,
            coe_training_duration,
            simulation_duration
        )

        self._moving_average_window_size_seconds = moving_average_window_size_seconds

    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        start_time_for_moving_average = (self._start_time_simulation_timestamp - self._moving_average_window_size_seconds) * 1000
        simulation_end_time_timestamp_milliseconds = (self._end_time_simulation_timestamp + 1) * 1000

        first_timestamp_for_last_event = self.get_last_value(
            self._orderbook_df['Timestamp'].values,
            start_time_for_moving_average
        )
        last_timestamp_for_next_event = self.get_next_value(
            self._orderbook_df['Timestamp'].values,
            simulation_end_time_timestamp_milliseconds
        )
        last_time_required_for_moving_average = (
            self._end_time_simulation_timestamp + self._moving_average_window_size_seconds
        ) * 1000

        last_timestamp = (
            last_timestamp_for_next_event
            if last_timestamp_for_next_event > last_time_required_for_moving_average
            else last_time_required_for_moving_average
        )
        
        testing_df = self._orderbook_df[
            (self._orderbook_df['Timestamp'] >= first_timestamp_for_last_event) &
            (self._orderbook_df['Timestamp'] <= last_timestamp)
        ].copy()
        testing_df = testing_df[['Timestamp', 'BaseImbalance', 'Return']]
        testing_df["DeltaT"] = testing_df["Timestamp"].shift(-1) - testing_df["Timestamp"]

        moving_average_window_size_milliseconds = self._moving_average_window_size_seconds * 1000

        df_map = {
            'Timestamp': list(),
            'NearestEventTimestampNotScaled': list(),
            'LastEventTimestampNotScaled': list(),
            'RealNextEventTimestampNotScaled': list(),
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
            df_map['NearestEventTimestampNotScaled'].append(nearest_timestamp)
            last_timestamp = self.get_last_value(testing_df['Timestamp'].values, i)
            df_map['LastEventTimestampNotScaled'].append(last_timestamp)
            real_next_timestamp = self.get_next_value(testing_df['Timestamp'].values, i)
            df_map['RealNextEventTimestampNotScaled'].append(real_next_timestamp)

        moving_average_df = pd.DataFrame(df_map)

        return moving_average_df


