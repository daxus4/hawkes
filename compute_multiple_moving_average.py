import json
import os
from typing import List

import numpy as np
import pandas as pd

ORDERBOOK_DIRECTORY_PATH = '/home/davide/Desktop/phd/bitfinex-api-py/data/orderbook_changes/'
RESULTS_DIRECTORY_PATH = '/home/davide/Desktop/phd/hawkes/data_moving_average'
DENSITY_FILE_PATH = '/home/davide/Desktop/phd/hawkes/data/file_densities_map.json'

SIMULATION_DURATION = pd.Timedelta(minutes=2, seconds=0)
COE_TRAINING_DURATION = pd.Timedelta(minutes=30, seconds=0)
MOVING_AVERAGE_WINDOW_SIZES_SECONDS = [30, 60, 120, 180, 240]
BASE_IMBALANCE_ORDERBOOK_LEVEL = 8

def get_preprocessed_df(df: pd.DataFrame, base_imbalance_orderbook_level: int) -> pd.DataFrame:
    df['MidPrice'] = (df["AskPrice1"]+df["BidPrice1"])/2
    df['Return'] = (-df["MidPrice"]+df["MidPrice"].shift(-1)) / df["MidPrice"]

    pbid = df["BidPrice1"] - df[f"BidPrice{base_imbalance_orderbook_level}"]
    pask = df[f"AskPrice{base_imbalance_orderbook_level}"] - df["AskPrice1"]
    df["BaseImbalance"] = (pbid-pask)/(pbid+pask)

    df=df.dropna(subset=['Return', 'BaseImbalance'])
    df = df[df['Return'] != 0]

    return df

def get_coe_training_df(
    df: pd.DataFrame,
    coe_training_start_time_timestamp_seconds: float,
    coe_training_end_time_timestamp_seconds: float
) -> pd.DataFrame:
    coe_training_start_time_timestamp_milliseconds = coe_training_start_time_timestamp_seconds * 1000
    coe_training_end_time_timestamp_milliseconds = coe_training_end_time_timestamp_seconds * 1000
    df = df[
        (df['Timestamp'] >= coe_training_start_time_timestamp_milliseconds) & 
        (df['Timestamp'] < coe_training_end_time_timestamp_milliseconds)
    ].copy()
    df = df[['Timestamp', 'BaseImbalance', 'Return']].copy()
    df['Timestamp'] = (df['Timestamp'] - coe_training_start_time_timestamp_milliseconds) / 1000
    df = df.dropna()

    return df

def get_coe_testing_df_moving_average(
    orderbook_df: pd.DataFrame,
    coe_training_start_time_timestamp_seconds: float,
    start_time_simulation_timestamp: float,
    simulation_end_time_timestamp_seconds: float,
    moving_average_window_size_seconds: float,
) -> pd.DataFrame:
    start_time_for_moving_average = (start_time_simulation_timestamp - moving_average_window_size_seconds) * 1000
    testing_df = orderbook_df[
        (orderbook_df['Timestamp'] >= start_time_for_moving_average) &
        (orderbook_df['Timestamp'] <= (
            simulation_end_time_timestamp_seconds + moving_average_window_size_seconds
        ) * 1000)
    ].copy()
    testing_df = testing_df[['Timestamp', 'BaseImbalance', 'Return']]
    testing_df["DeltaT"] = testing_df["Timestamp"].shift(-1) - testing_df["Timestamp"]

    moving_average_window_size_milliseconds = moving_average_window_size_seconds * 1000

    df_map = {
        'Timestamp': list(),
        'TimestampReal': list(),
        'PredictionError': list(),
    }

    for i in range(int(start_time_simulation_timestamp * 1000), int(end_time_simulation_timestamp * 1000), 1000):        
        moving_average = testing_df[
            (testing_df['Timestamp'] >= (i - moving_average_window_size_milliseconds)) &
            (testing_df['Timestamp'] <= i)
        ]['DeltaT'].mean()
        moving_average = int(moving_average) if not np.isnan(moving_average) else moving_average_window_size_milliseconds

        predicted_timestamp = i + moving_average
        df_map['Timestamp'].append(predicted_timestamp)
        nearest_timestamp = get_nearest_value(testing_df['Timestamp'].values, predicted_timestamp)
        df_map['TimestampReal'].append(nearest_timestamp)
    
    moving_average_df = pd.DataFrame(df_map)
    moving_average_df = pd.merge(moving_average_df, testing_df, left_on='TimestampReal', right_on='Timestamp', suffixes=('', '_y'))
    moving_average_df = moving_average_df.drop(columns=['TimestampReal', 'DeltaT', 'Timestamp_y'])
    moving_average_df['Timestamp'] = (moving_average_df['Timestamp']/1000 - coe_training_start_time_timestamp_seconds)
    moving_average_df['PredictionError'] = moving_average_df['PredictionError']/1000
    return moving_average_df

def get_nearest_value(array: np.ndarray, value: float) -> float:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_complete_coe_df(training_df: pd.DataFrame, testing_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([training_df, testing_df])
    df['Timestamp'] = df['Timestamp'].round(3)
    df = df.reset_index(drop=True)
    df.columns = ['1', '2', '3']
    return df


def save_json_file(directory: str, filename_timestamp, max_density_start_time_simulation: str, json_data, suffix: str):
    filename_path = os.path.join(
        directory,
        filename_timestamp + '_' + max_density_start_time_simulation + f'_{suffix}.json'
    )

    with open(filename_path, 'w') as json_file:
        json.dump(json_data, json_file)

def save_complete_df(
    directory: str,
    filename_timestamp: str,
    max_density_start_time_simulation: str,
    coe_training_df: pd.DataFrame,
    coe_testing_df: pd.DataFrame,
    suffix: str
):
    filename_path = os.path.join(
        directory,
        filename_timestamp + '_' + max_density_start_time_simulation + f'_{suffix}.csv'
    )
    
    complete_moving_average_coe_df = get_complete_coe_df(
        coe_training_df, coe_testing_df
    )

    complete_moving_average_coe_df.to_csv(filename_path, index=False)

def get_index_positions_without_old_greater_predictions(series: pd.Series) -> pd.Series:
    to_delete_index_positions = []
    for i in range(0, len(series)):
        for j in range(i+1, len(series)):
            if series.iloc[i] > series.iloc[j]:
                to_delete_index_positions.append(i)
                break
            
    return to_delete_index_positions

def get_dataframe_without_index_positions(df: pd.DataFrame, index_positions: List) -> pd.DataFrame:
    return df.drop(df.index[index_positions])


if __name__ == '__main__':
    simulation_period_duration_seconds = int(SIMULATION_DURATION.total_seconds())
    coe_training_duration_seconds = COE_TRAINING_DURATION.total_seconds()

    with open(DENSITY_FILE_PATH) as json_file:
        file_densities_map = json.load(json_file)

    best_densities_file = pd.read_csv('/home/davide/Desktop/phd/hawkes/data/best_densities_full.csv')
    best_densities_timestamps = best_densities_file['timestamp'].values.tolist()
    best_densities_timestamps = [str(x) for x in best_densities_timestamps]
    best_densities_hours = best_densities_file['timestamp_density'].values.tolist()

    # getting already computed files
    already_computed_file_timestamps = set()
    for filename in os.listdir(RESULTS_DIRECTORY_PATH):
        if filename.endswith('.csv'):
            already_computed_file_timestamps.add(filename.split('_')[0])

    for filename, densities_info in file_densities_map.items():
        filename_timestamp = filename.split('_')[2].split('.')[0]
        if filename_timestamp in already_computed_file_timestamps:
            continue

        if filename_timestamp not in best_densities_timestamps:
            continue

        filename_path = os.path.join(ORDERBOOK_DIRECTORY_PATH, filename)
        for density_info in densities_info:
            if int(pd.Timestamp(density_info[0]).strftime('%H%M%S')) not in best_densities_hours:
                continue

            max_density_start_time_simulation = pd.Timestamp(density_info[0])
            max_density_start_time_simulation_str = max_density_start_time_simulation.strftime('%H%M%S')

            start_time_simulation_timestamp = max_density_start_time_simulation.timestamp()

            end_time_simulation = max_density_start_time_simulation + SIMULATION_DURATION
            end_time_simulation_timestamp = end_time_simulation.timestamp()

            coe_training_start_time = max_density_start_time_simulation - COE_TRAINING_DURATION
            coe_training_start_time_timestamp = coe_training_start_time.timestamp()
            
            try:
                orderbook_df = pd.read_csv(filename_path, sep='\t')
                orderbook_df = get_preprocessed_df(orderbook_df, BASE_IMBALANCE_ORDERBOOK_LEVEL)

                coe_training_df = get_coe_training_df(
                    orderbook_df, coe_training_start_time_timestamp, start_time_simulation_timestamp
                )

                for moving_average_window_size_seconds in MOVING_AVERAGE_WINDOW_SIZES_SECONDS:
                    moving_average_testing_coe_df = get_coe_testing_df_moving_average(
                        orderbook_df,
                        coe_training_start_time_timestamp,
                        start_time_simulation_timestamp,
                        end_time_simulation_timestamp,
                        moving_average_window_size_seconds
                    )

                    ma_testing_coe_index_positions = get_index_positions_without_old_greater_predictions(
                        moving_average_testing_coe_df['Timestamp']
                    )
                    moving_average_testing_coe_df = get_dataframe_without_index_positions(
                        moving_average_testing_coe_df, ma_testing_coe_index_positions
                    )


                    prediction_errors = moving_average_testing_coe_df['PredictionError'].values.tolist()
                    moving_average_testing_coe_df.drop(columns=['PredictionError'], inplace=True)

                    save_json_file(
                        RESULTS_DIRECTORY_PATH,
                        filename_timestamp,
                        max_density_start_time_simulation_str,
                        prediction_errors,
                        f'{moving_average_window_size_seconds}_prediction_errors'
                    ) 

                    save_complete_df(
                        RESULTS_DIRECTORY_PATH,
                        filename_timestamp,
                        max_density_start_time_simulation_str,
                        coe_training_df,
                        moving_average_testing_coe_df,
                        f"{moving_average_window_size_seconds}_BI{BASE_IMBALANCE_ORDERBOOK_LEVEL}"
                    )
                
            except Exception as e:
                print(f'Error while computing {filename_timestamp} with density {density_info[0]}: {e}')


