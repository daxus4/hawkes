import os
from typing import Dict, List

import pandas as pd

START_TEST_TIMESTAMP = 1800

def get_prefix_df_filepath_map(directory: str) -> Dict[str, Dict[str, List[str]]]:
    prefix_df_filename_map = dict()

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            prefix = '_'.join(filename.split("_")[0:2])
            if prefix not in prefix_df_filename_map:
                prefix_df_filename_map[prefix] = dict()
            
            method = filename.split("_")[2]
            prefix_df_filename_map[prefix][method] = os.path.join(directory,filename)
    
    prefix_df_filename_map = {prefix: method_df_map for prefix, method_df_map in prefix_df_filename_map.items() if len(method_df_map) == 4}

    return prefix_df_filename_map

def get_prefix_df_map(prefix_df_filepath_map: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    prefix_df_map = dict()

    for prefix, df_filepath_map in prefix_df_filepath_map.items():
        prefix_df_map[prefix] = {
            'hawkes': read_event_df(df_filepath_map['hawkes']),
            'movingaverage': read_event_df(df_filepath_map['movingaverage']),
            'oracle': read_event_df(df_filepath_map['oracle']),
            'naive': read_event_df(df_filepath_map['naive']),
        }

    return prefix_df_map

def read_event_df(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, header=0)

def get_minimun_index_length(dfs: List[pd.DataFrame]) -> int:
    return min([len(df.index) for df in dfs])

def get_test_df(df: pd.DataFrame, start_test_timestamp: int) -> pd.DataFrame:
    return df[df['1'] >= start_test_timestamp].copy()

def get_train_df(df: pd.DataFrame, end_training_timestamp: int) -> pd.DataFrame:
    return df[df['1'] < end_training_timestamp].copy()

if __name__ == '__main__':
    timestamp_prefix_min_index_length_map = dict()
    for event_df_directory in [f'data_{i}min' for i in [5,10,15,20]]:

        prefix_df_filepath_map = get_prefix_df_filepath_map(event_df_directory)
        prefix_df_map = get_prefix_df_map(prefix_df_filepath_map)

        prefix_sampled_df_map = dict()

        for timestamp_prefix, method_df_map in prefix_df_map.items():
            min_index_length = get_minimun_index_length([get_test_df(df, START_TEST_TIMESTAMP) for df in method_df_map.values()])

            if timestamp_prefix not in timestamp_prefix_min_index_length_map:
                timestamp_prefix_min_index_length_map[timestamp_prefix] = min_index_length
            else:
                timestamp_prefix_min_index_length_map[timestamp_prefix] = min(
                    timestamp_prefix_min_index_length_map[timestamp_prefix],
                    min_index_length
                )

    for event_df_directory in [f'data_{i}min' for i in [5,10,15,20]]:
        prefix_df_filepath_map = get_prefix_df_filepath_map(event_df_directory)
        prefix_df_map = get_prefix_df_map(prefix_df_filepath_map)

        for timestamp_prefix, method_df_map in prefix_df_map.items():

            method_sampled_df_map = dict()

            for method, df in method_df_map.items():
                train_df = get_train_df(df, START_TEST_TIMESTAMP)
                test_df = get_test_df(df, START_TEST_TIMESTAMP)
                method_sampled_df_map[method] = pd.concat(
                    [
                        train_df,
                        test_df.sample(
                            n=timestamp_prefix_min_index_length_map[
                                timestamp_prefix
                            ]
                        ).sort_values(by='1')
                    ]
                )

            prefix_sampled_df_map[timestamp_prefix] = method_sampled_df_map

        sampled_dir = event_df_directory + '_sampled'

        if not os.path.exists(sampled_dir):
            os.makedirs(event_df_directory + '_sampled')

        for timestamp_prefix, method_df_map in prefix_sampled_df_map.items():
            for method, df in method_df_map.items():
                df.to_csv(os.path.join(sampled_dir, f"{timestamp_prefix}_{method}.csv"), index=False)