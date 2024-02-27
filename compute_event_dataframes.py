import os
from typing import List, Optional

import pandas as pd
import yaml

from constants import DATA_DIRECTORY, HAWKES_METHOD_STR, MOVING_AVERAGE_METHOD_STR

CONF_FILE_PATH = 'data/conf_event_df.yml'


def get_conf(path: str):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf

def create_method_folders(
    path: str,
    methods: List[str],
    hawkes_training_times: Optional[List[int]] = None,
    moving_average_training_times: Optional[List[int]] = None,
):
    for method in methods:
        if method == HAWKES_METHOD_STR:
            os.makedirs(
                os.path.join(path, 'hawkes_best_decays'), exist_ok=True
            )

            for training_time in hawkes_training_times:
                _create_folder_for_submethod(path, method, training_time)

        elif method == MOVING_AVERAGE_METHOD_STR:
            for training_time in moving_average_training_times:
                _create_folder_for_submethod(path, method, training_time)
        else:
            os.makedirs(
                os.path.join(path, method), exist_ok=True
            )

def _create_folder_for_submethod(path: str, method: str, training_time: str):
    os.makedirs(
        os.path.join(path, f'{method}_{training_time}'), exist_ok=True
    )

def get_densities_df(path: str) -> pd.DataFrame:
    best_densities_df = pd.read_csv(path)
    best_densities_df['timestamp_density'] = best_densities_df['timestamp_density'].astype(str).str.zfill(6)
    best_densities_df = best_densities_df[['timestamp', 'timestamp_density']].groupby('timestamp').agg({'timestamp_density': list}).reset_index()
    return best_densities_df

def get_already_computed_file_prefixes(path: str) -> List[str]:
    return [
        '_'.join(file.split('_')[0:1]) for file in os.listdir(path) if file.endswith('.csv')
    ]

def get_orderbook_df_file_path(
    orderbook_dfs_path: str,
    timestamp: str,
) -> str:
    filename_for_finished_orderbook = f'orderbook_changes_{timestamp}.tsv'

    if filename_for_finished_orderbook in os.listdir(orderbook_dfs_path):
        return os.path.join(
            orderbook_dfs_path,
            filename_for_finished_orderbook
        )
    else:
        filename_for_interrupted_orderbook = f'orderbook_changes_{timestamp}_interrupted.tsv'
        if filename_for_interrupted_orderbook in os.listdir(orderbook_dfs_path):
            return os.path.join(
                orderbook_dfs_path,
                filename_for_interrupted_orderbook
            )
        else:
            raise FileNotFoundError(f'No file found for timestamp {timestamp}')

if __name__ == '__main__':
    config_map = get_conf(CONF_FILE_PATH)
    
    orderbook_dfs_path = config_map['orderbook_dfs_path']
    densities_file_path = config_map['densities_file_path']
    methods = config_map['method']

    hawkes_training_times = config_map['hawkes_training_time'] if HAWKES_METHOD_STR in methods else None
    moving_average_training_times = config_map['movingaverage_training_time'] if MOVING_AVERAGE_METHOD_STR in methods else None

    # create_method_folders(
    #     DATA_DIRECTORY,
    #     methods,
    #     hawkes_training_times,
    #     moving_average_training_times,
    # )

    best_densities_df = get_densities_df(densities_file_path)
    
    for row in best_densities_df.itertuples():
        timestamp_file = row.timestamp
        
        for hours_start_simulation in row.timestamp_density:
            
