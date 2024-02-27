import pandas as pd

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class NaiveEventDatabaseExtractor(EventDatabaseExtractor):
    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        start_time_simulation_timestamp_milliseconds = self._start_time_simulation_timestamp * 1000
        simulation_end_time_timestamp_milliseconds = (self._end_time_simulation_timestamp + 1) * 1000
        df = self._orderbook_df[
            (self._orderbook_df['Timestamp'] >= start_time_simulation_timestamp_milliseconds) &
            (self._orderbook_df['Timestamp'] < simulation_end_time_timestamp_milliseconds)
        ].copy()
        testing_df_map = {
            'Timestamp': [],
            'RealTimestamp': [],
        }

        for timestamp in range(
            int(start_time_simulation_timestamp_milliseconds),
            int(simulation_end_time_timestamp_milliseconds),
            1000
        ):
            testing_df_map['Timestamp'].append(timestamp)
            testing_df_map['RealTimestamp'].append(self.get_nearest_value(df['Timestamp'], timestamp))
        
        testing_df = pd.DataFrame(testing_df_map)
        testing_df['Timestamp'] = (testing_df['Timestamp']/ 1000) - self._coe_training_start_time_timestamp
        testing_df['RealTimestampNotScaled'] = testing_df['RealTimestamp']

        return testing_df

