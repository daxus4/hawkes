import pandas as pd

from event_database_extractor.event_database_extractor import EventDatabaseExtractor


class OracleEventDatabaseExtractor(EventDatabaseExtractor):
    def _get_coe_timestamp_test_df(self) -> pd.DataFrame:
        testing_df = self._orderbook_df[
            (self._orderbook_df['Timestamp'] >= self._start_time_simulation_timestamp * 1000) &
            (self._orderbook_df['Timestamp'] < self._end_time_simulation_timestamp * 1000)
        ][['Timestamp']].copy()

        # keep only the first row for each second
        testing_df['SecondTimestamp'] = testing_df['Timestamp'] // 1000
        testing_df = testing_df.groupby('SecondTimestamp').first().reset_index()

        testing_df.drop(columns=['SecondTimestamp'], inplace=True)
        testing_df['NearestEventTimestampNotScaled'] = testing_df['Timestamp']
        testing_df['LastEventTimestampNotScaled'] = testing_df['Timestamp']
        testing_df['RealNextEventTimestampNotScaled'] = testing_df['Timestamp']

        return testing_df
