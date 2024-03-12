import pandas as pd

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class NaiveEventDatabaseExtractor(EventDatabaseExtractor):
    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        start_time_simulation_timestamp_milliseconds = self._start_time_simulation_timestamp * 1000
        simulation_end_time_timestamp_milliseconds = (self._end_time_simulation_timestamp + 1) * 1000

        first_timestamp_for_last_event = self.get_last_value(
            self._orderbook_df['Timestamp'].values,
            start_time_simulation_timestamp_milliseconds - 1000
        )
        last_timestamp_for_next_event = self.get_next_value(
            self._orderbook_df['Timestamp'].values,
            simulation_end_time_timestamp_milliseconds
        )

        df = self._orderbook_df[
            (self._orderbook_df['Timestamp'] >= first_timestamp_for_last_event) &
            (self._orderbook_df['Timestamp'] < last_timestamp_for_next_event)
        ].copy()

        testing_df_map = {
            'Timestamp': [],
            'NearestEventTimestampNotScaled': [],
            'LastEventTimestampNotScaled': [],
            'RealNextEventTimestampNotScaled': [],
        }

        for timestamp in range(
            int(start_time_simulation_timestamp_milliseconds),
            int(simulation_end_time_timestamp_milliseconds),
            1000
        ):
            testing_df_map['Timestamp'].append(timestamp)
            testing_df_map['NearestEventTimestampNotScaled'].append(
                self.get_nearest_value(df['Timestamp'], timestamp)
            )
            testing_df_map['LastEventTimestampNotScaled'].append(
                self.get_last_value(
                    self._orderbook_df['Timestamp'].values, timestamp - 1000
                )
            )
            testing_df_map['RealNextEventTimestampNotScaled'].append(
                self.get_next_value(
                    self._orderbook_df['Timestamp'].values, timestamp - 1000
                )
            )

        
        testing_df = pd.DataFrame(testing_df_map)

        return testing_df

