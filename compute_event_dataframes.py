import os
from typing import Dict, List, Optional

import pandas as pd
import yaml

from constants import (
    DATA_DIRECTORY,
    HAWKES_METHOD_STR,
    MOVING_AVERAGE_METHOD_STR,
    NAIVE_METHOD_STR,
    ORACLE_METHOD_STR,
)
from event_database_extractor.hawkes_event_database_extractor import (
    HawkesEventDatabaseExtractor,
)
from event_database_extractor.movingaverage_event_database_extractor import (
    MovingAverageEventDatabaseExtractor,
)
from event_database_extractor.naive_event_database_extractor import (
    NaiveEventDatabaseExtractor,
)
from event_database_extractor.oracle_event_database_extractor import (
    OracleEventDatabaseExtractor,
)
from method import Method

CONF_FILE_PATH = 'data/conf_event_df.yml'

SIMULATION_DURATION = pd.Timedelta(minutes=2, seconds=0)
COE_TRAINING_DURATION = pd.Timedelta(minutes=30, seconds=0)

HAWKES_PREDICTION_PERIOD_DURATION = pd.Timedelta(seconds=30)
HAWKES_WARM_UP_PERIOD_DURATION = pd.Timedelta(minutes=2, seconds=30)

def get_conf(path: str):
    with open(path, 'r') as f:
        conf = yaml.safe_load(f)
    return conf

def create_method_folders(
    path: str,
    methods: List[Method],
):
    for method in methods:
        if method.name == HAWKES_METHOD_STR:
            os.makedirs(
                os.path.join(path, 'hawkes_best_decays'), exist_ok=True
            )

        for submethod_folder in method.get_submethods_folders():
            _create_folder_for_submethod(path, submethod_folder)

def _create_folder_for_submethod(path: str, submethod: str):
    os.makedirs(
        os.path.join(path, submethod), exist_ok=True
    )

def get_densities_df(path: str) -> pd.DataFrame:
    best_densities_df = pd.read_csv(path)
    best_densities_df = best_densities_df[['timestamp', 'timestamp_density']].groupby('timestamp').agg({'timestamp_density': list}).reset_index()
    return best_densities_df

def get_submethod_already_computed_files_map(
    path: str,
    methods: List[Method],
) -> Dict[str, List[str]]:
    submethod_already_computed_file_prefixes_map = {}
    for method in methods:
        for submethod_folder in method.get_submethods_folders():
            submethod_already_computed_file_prefixes_map[submethod_folder] = _get_already_computed_files(
                os.path.join(path, submethod_folder)
            )

    return submethod_already_computed_file_prefixes_map

def _get_already_computed_files(path: str) -> List[str]:
    return [
        file for file in os.listdir(path) if file.endswith('.csv')
    ]

def _get_coe_df_filename(
    timestamp: int,
    timestamp_start_simulation: str,
    base_imbalance_level: int,
) -> str:
    return f'{timestamp}_{timestamp_start_simulation}_BI{base_imbalance_level}.csv'


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


def get_preprocessed_df(df: pd.DataFrame, base_imbalance_orderbook_level: int) -> pd.DataFrame:
    df['MidPrice'] = (df["AskPrice1"]+df["BidPrice1"])/2
    df['Return'] = (-df["MidPrice"]+df["MidPrice"].shift(-1)) / df["MidPrice"]

    pbid = df["BidPrice1"] - df[f"BidPrice{base_imbalance_orderbook_level}"]
    pask = df[f"AskPrice{base_imbalance_orderbook_level}"] - df["AskPrice1"]
    df["BaseImbalance"] = (pbid-pask)/(pbid+pask)

    df=df.dropna(subset=['Return', 'BaseImbalance'])
    df = df[df['Return'] != 0]
    df = df[['Timestamp', 'BaseImbalance', 'Return']]

    return df


def get_starting_event_df(
    orderbook_dfs_path: str, base_imbalance_level: int, timestamp_file: int
) -> pd.DataFrame:
    orderbook_df_file_path = get_orderbook_df_file_path(
                        orderbook_dfs_path,
                        timestamp_file,
                    )
    orderbook_df = pd.read_csv(orderbook_df_file_path, sep='\t')
    orderbook_df = get_preprocessed_df(orderbook_df, base_imbalance_level)
    return orderbook_df


def _get_hawkes_submethod_value_decay_df_map(
    hawkes_parameters: List[int]
) -> Dict[str, pd.DataFrame]:
    hawkes_submethod_value_decay_df_map = {}
    for parameter in hawkes_parameters:
        df_path = os.path.join(
            DATA_DIRECTORY,
            'hawkes_best_decays',
            f'hawkes_decay_{parameter}min.tsv'
        )

        if not os.path.exists(df_path):
            hawkes_submethod_value_decay_df_map[parameter] = pd.DataFrame(
                columns=['timestamp', 'timestamp_density', 'decay']
            )
        else:
            hawkes_submethod_value_decay_df_map[parameter] = pd.read_csv(
                df_path, sep='\t'
            )

    return hawkes_submethod_value_decay_df_map


def save_new_decay(
    decay_df: pd.DataFrame,
    submethod_value: int,
) -> None:
    decay_df.to_csv(
        os.path.join(
            DATA_DIRECTORY,
            'hawkes_best_decays',
            f'hawkes_decay_{submethod_value}min.tsv'
        ),
        sep='\t',
        index=False,
    )

def get_best_decay(
    hawkes_submethod_value_decay_df_map: Dict[str, pd.DataFrame],
    timestamp_file: int,
    timestamp_start_simulation: int,
    submethod_value: int
) -> Optional[float]:
    df = hawkes_submethod_value_decay_df_map[submethod_value]

    row = df.loc[
        (df['timestamp'] == timestamp_file) &
        (df['timestamp_density'] == timestamp_start_simulation),
    ]

    if row.empty:
        return None
    return row['decay'].values[0]

if __name__ == '__main__':
    config_map = get_conf(CONF_FILE_PATH)
    
    orderbook_dfs_path = config_map['orderbook_dfs_path']
    densities_file_path = config_map['densities_file_path']
    method_names = config_map['methods']
    base_imbalance_level = config_map['base_imbalance_level']

    methods: List[Method] = [
        Method.from_conf(method, config_map) for method in method_names
    ]

    create_method_folders(DATA_DIRECTORY, methods)

    submethod_already_computed_file_prefixes_map = get_submethod_already_computed_files_map(
        DATA_DIRECTORY,
        methods,
    )

    best_densities_df = get_densities_df(densities_file_path)

    if HAWKES_METHOD_STR in method_names:
        hawkes_prediction_period_duration_seconds = HAWKES_PREDICTION_PERIOD_DURATION.total_seconds()
        hawkes_warm_up_period_duration_seconds = HAWKES_WARM_UP_PERIOD_DURATION.total_seconds()
        hawkes_submethod_value_decay_df_map = _get_hawkes_submethod_value_decay_df_map(
            config_map['hawkes_parameters']
        )

    for row in best_densities_df.itertuples():
        timestamp_file = row.timestamp
        
        for timestamp_start_simulation in row.timestamp_density:
            for method in methods:
                for submethod_folder, submethod_value in zip(
                    method.get_submethods_folders(),
                    method.get_submethods_values(),
                ):
                    completed_coe_df_filename = _get_coe_df_filename(
                        timestamp_file,
                        timestamp_start_simulation,
                        base_imbalance_level,
                    )
                    if completed_coe_df_filename in submethod_already_computed_file_prefixes_map[submethod_folder]:
                        continue

                    orderbook_df = get_starting_event_df(
                        orderbook_dfs_path, base_imbalance_level, timestamp_file
                    )

                    start_time_simulation = pd.Timestamp(timestamp_start_simulation, unit='s')

                    if method.name == HAWKES_METHOD_STR:
                        best_decay = get_best_decay(
                            hawkes_submethod_value_decay_df_map,
                            timestamp_file,
                            timestamp_start_simulation,
                            submethod_value
                        )

                        event_database_extractor = HawkesEventDatabaseExtractor(
                            orderbook_df,
                            start_time_simulation,
                            COE_TRAINING_DURATION,
                            SIMULATION_DURATION,
                            hawkes_training_duration=pd.Timedelta(minutes=submethod_value),
                            prediction_period_duration=HAWKES_PREDICTION_PERIOD_DURATION,
                            warm_up_period_duration=HAWKES_WARM_UP_PERIOD_DURATION,
                            best_decay=best_decay,
                        )

                    if method.name == MOVING_AVERAGE_METHOD_STR:
                        event_database_extractor = MovingAverageEventDatabaseExtractor(
                            orderbook_df,
                            start_time_simulation,
                            COE_TRAINING_DURATION,
                            SIMULATION_DURATION,
                            moving_average_window_size_seconds=submethod_value,
                        )

                    if method.name == NAIVE_METHOD_STR:
                        event_database_extractor = NaiveEventDatabaseExtractor(
                            orderbook_df,
                            start_time_simulation,
                            COE_TRAINING_DURATION,
                            SIMULATION_DURATION,
                        )
                    
                    if method.name == ORACLE_METHOD_STR:
                        event_database_extractor = OracleEventDatabaseExtractor(
                            orderbook_df,
                            start_time_simulation,
                            COE_TRAINING_DURATION,
                            SIMULATION_DURATION,
                        )

                    complete_coe_df = event_database_extractor.get_complete_coe_df()

                    complete_coe_df.to_csv(
                        os.path.join(DATA_DIRECTORY, submethod_folder, completed_coe_df_filename),
                        index=False,
                    )

                    if method.name == HAWKES_METHOD_STR:
                        new_decay = event_database_extractor.best_decay
                        decay_df = hawkes_submethod_value_decay_df_map[submethod_value]

                        row = decay_df.loc[
                            (decay_df['timestamp'] == timestamp_file) &
                            (decay_df['timestamp_density'] == timestamp_start_simulation)
                        ]

                        if row.empty:
                            decay_df = decay_df.append(
                                {
                                    'timestamp': timestamp_file,
                                    'timestamp_density': timestamp_start_simulation,
                                    'decay': new_decay,
                                },
                                ignore_index=True,
                            )
                        else:
                            decay_df.loc[
                                (decay_df['timestamp'] == timestamp_file) &
                                (decay_df['timestamp_density'] == timestamp_start_simulation),
                                'decay'
                            ] = new_decay

                        save_new_decay(
                            decay_df,
                            submethod_value,
                        )

                    print(f'Finished {method.name} {submethod_value} {completed_coe_df_filename}')                