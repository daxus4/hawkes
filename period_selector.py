import json
import os
from typing import Dict, List, Tuple

import pandas as pd

SIMULATION_TIME_DURATION = pd.Timedelta('2min')
STARTING_TIME_OFFSET = pd.Timedelta('30min')
DISTANCE_TIME_FOR_MAX_DENSITY = pd.Timedelta('35min')
PATH_ORDERBOOK_DIRECTORY = '/home/davide/Desktop/phd/bitfinex-api-py/data/{}/orderbook_changes/'
MARKETS = ['BTC_USDT', 'ETH_BTC', 'ETH_USD', 'ETH_USDT']

def get_rounded_timestamp_series(timestamp_series: pd.Series, round_to: str):
    return timestamp_series.dt.round(round_to)
        
def get_group_density_df(value_series: pd.Series, grouping_lenght: pd.Timedelta) -> pd.DataFrame:
    densities = list()

    for timestamp in value_series.values:
        ending_timestamp = timestamp + grouping_lenght
        density = len(value_series[(value_series >= timestamp) & (value_series < ending_timestamp)])

        densities.append(density)

    return pd.DataFrame({"timestamp": value_series.values, "density": densities})

def get_timestamp_grouped_event_series(df: pd.DataFrame, col_name: str, round_to: str) -> pd.Series:
    grouped_series = get_rounded_timestamp_series(df[col_name], round_to)
    grouped_series = grouped_series.unique()
    grouped_series.sort()
    grouped_series = pd.Series(grouped_series)
    return grouped_series

def get_density_df(event_df: pd.DataFrame, col_name: str, grouping_lenght: pd.Timedelta, round_to: str) -> pd.DataFrame:
    timestamp_grouped_series = get_timestamp_grouped_event_series(event_df, col_name, round_to)
    timestamp_grouped_df = get_group_density_df(timestamp_grouped_series, grouping_lenght)
    return timestamp_grouped_df

def _get_density_df_row_distanced_from_timestamp(timestamp: pd.Timestamp, density_df: pd.DataFrame, min_distance: pd.Timedelta) -> pd.DataFrame:
    return density_df[
        (density_df['timestamp'] <= (timestamp - min_distance)) |
        (density_df['timestamp'] >= (timestamp + min_distance))
    ]

def get_local_max_density_groups_distanced(density_df: pd.DataFrame, min_distance: pd.Timedelta) -> List[Tuple[pd.Timestamp, int]]:
    max_density_groups = density_df[density_df['density'] == density_df['density'].max()]
    current_max_density_timestamp = max_density_groups['timestamp'].iloc[0]
    current_max_density = max_density_groups['density'].iloc[0]

    max_density_timestamps = [(current_max_density_timestamp.strftime("%Y-%m-%d %H:%M:%S"), str(current_max_density))]

    remaining_max_density_groups = _get_density_df_row_distanced_from_timestamp(
        current_max_density_timestamp, density_df, min_distance
    )

    while not remaining_max_density_groups.empty:
        max_density_groups = remaining_max_density_groups[
            remaining_max_density_groups['density'] == remaining_max_density_groups['density'].max()
        ]
        current_max_density_timestamp = max_density_groups['timestamp'].iloc[0]
        current_max_density = max_density_groups['density'].iloc[0]

        max_density_timestamps.append((current_max_density_timestamp.strftime("%Y-%m-%d %H:%M:%S"), str(current_max_density)))

        remaining_max_density_groups = _get_density_df_row_distanced_from_timestamp(
            current_max_density_timestamp, remaining_max_density_groups, min_distance
        )
    
    return max_density_timestamps

def get_preprocessed_orderbook_df(orderbook_df: pd.DataFrame, initial_offset: pd.Timedelta, final_offset: pd.Timedelta) -> pd.DataFrame:
    orderbook_df['Timestamp'] = pd.to_datetime(orderbook_df['Timestamp'], unit='ms')

    min_correct_timestamp = orderbook_df['Timestamp'].min() + initial_offset
    max_correct_timestamp = orderbook_df['Timestamp'].max() - final_offset

    orderbook_df = orderbook_df[
        (orderbook_df['Timestamp'] >= min_correct_timestamp) &
        (orderbook_df['Timestamp'] <= max_correct_timestamp)
    ].copy()

    orderbook_df['MidPrice'] = (orderbook_df["AskPrice1"]+orderbook_df["BidPrice1"])/2
    orderbook_df['Difference'] = (-orderbook_df["MidPrice"]+orderbook_df["MidPrice"].shift(-1))
    orderbook_df = orderbook_df.dropna()
    orderbook_df = orderbook_df[orderbook_df['Difference'] != 0]

    orderbook_df['timestamp_rounded'] = get_rounded_timestamp_series(orderbook_df['Timestamp'], '1s')
    return orderbook_df

def get_files(directory: str) -> List[str]:
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def save_densities_table(config: Dict, file_path: str):
    df_map = {
        'timestamp': [],
        'timestamp_density': [],
        'density': [],
    }

    for k, v in config.items():
        num_densities = len(v)
        timestamp = k.split('_')[2].split('.')[0]
        df_map['timestamp'].extend([float(timestamp)] * num_densities)

        for info_density in v:
            df_map['timestamp_density'].append(int(pd.Timestamp(info_density[0]).timestamp()))
            df_map['density'].append(int(info_density[1]))

    df_map = pd.DataFrame(df_map)
    df_map.sort_values(by=['density'], inplace=True, ascending=False)
    df_map['timestamp'] = df_map['timestamp'].apply(lambda x: int(x))

    df_map.to_csv(file_path, index=False)


if __name__ == "__main__":
    for market in MARKETS:
        path_orderbook_directory = PATH_ORDERBOOK_DIRECTORY.format(market)

        file_densities_map = dict()

        for orderbook_file_path in get_files(path_orderbook_directory):
            df_btc = pd.read_csv(os.path.join(path_orderbook_directory, orderbook_file_path), sep='\t')
            df_btc = get_preprocessed_orderbook_df(df_btc, STARTING_TIME_OFFSET, SIMULATION_TIME_DURATION)

            if not df_btc.empty:
                density_df = get_density_df(df_btc, 'Timestamp', SIMULATION_TIME_DURATION, '1s')
                file_densities_map[orderbook_file_path] = get_local_max_density_groups_distanced(density_df, DISTANCE_TIME_FOR_MAX_DENSITY)

        save_densities_table(file_densities_map, f'data_{market}/densities_table.csv')

        # save file_densities_map in json
        with open(f'data_{market}/file_densities_map.json', 'w') as f:
            json.dump(file_densities_map, f)