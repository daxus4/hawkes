import json
import os
from typing import List, Tuple
import tick.hawkes as hk
import numpy as np
import pandas as pd 

ORDERBOOK_DIRECTORY_PATH = '/home/davide/Desktop/phd/bitfinex-api-py/data/orderbook_changes/'
HAWKES_DIRECTORY_PATH = '/home/davide/Desktop/phd/hawkes/data_090min_training'
DENSITY_FILE_PATH = '/home/davide/Desktop/phd/hawkes/data/file_densities_map.json'

TRAINING_DURATION = pd.Timedelta(minutes=1, seconds=30)
SIMULATION_DURATION = pd.Timedelta(minutes=2, seconds=0)
PREDICTION_PERIOD_DURATION = pd.Timedelta(seconds=30)
WARM_UP_PERIOD_DURATION = pd.Timedelta(minutes=2, seconds=30)
COE_TRAINING_DURATION = pd.Timedelta(minutes=30, seconds=0)
MOVING_AVERAGE_WINDOW_SIZE_SECONDS = 120
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

def get_attack_times(timestamp_series: pd.Series, start_time_training: pd.Timestamp, end_time_simulation: pd.Timestamp) -> np.ndarray:
    attack_times = pd.to_datetime(timestamp_series, unit='ms').to_numpy()

    first_event_not_in_simulation = attack_times[attack_times > end_time_simulation][0]
    attack_times = attack_times[
        (attack_times >= start_time_training) & (attack_times <= end_time_simulation)
    ].copy()
    attack_times = np.append(attack_times, first_event_not_in_simulation)

    attack_times = np.array([dt.astype('datetime64[ms]').astype(float) / 1000 for dt in attack_times])

    return attack_times


def get_best_decays_with_cross_validation(
    attack_times: np.ndarray, min_beta: float = 0.01, max_beta: float = 100.0
) -> float:
    beta_values = np.linspace(min_beta, max_beta, num=3000)

    best_beta = None
    best_score = float('-inf')

    #  cross validation per trovare il miglior beta
    for beta in beta_values:
        decays = np.array([[beta]])
        hawkes_model = hk.HawkesExpKern(decays=decays)
        hawkes_model.fit([attack_times])
        score = hawkes_model.score([attack_times])
        if score > best_score:
            best_score = score
            best_beta = beta
    
    return best_beta


def get_hawkes_parameters_trained(
    attack_times: np.ndarray,
    decays: float = 0.06162109, 
) -> Tuple[float, List[np.ndarray], np.ndarray]:
    hawkes = hk.HawkesExpKern(decays=decays)
    hawkes.fit([attack_times])

    baseline = hawkes.baseline
    adjacency = hawkes.adjacency
    abb=baseline[0]
    baa= [np.array([decays])]
    caa=np.array([[adjacency[0,0]]])

    return abb, baa, caa

def get_hawkes_results(
    attack_times: np.ndarray,
    abb: float, baa: List[np.ndarray], caa:np.ndarray,
    simulation_period_duration_seconds: int,
    start_time_simulation_timestamp: float,
    warm_up_period_duration_seconds: float,
    prediction_period_duration_seconds: float,
) -> pd.DataFrame:
    simulation_results_map = {
        'start_time_simulation_timestamp': [],
        'real_next_event_timestamp': [],
        'predicted_next_event_timestamp': [],
        'prediction_error': [],
        'intensity': []
    }

    for i in range(simulation_period_duration_seconds):
        current_start_time_simulation_timestamp = start_time_simulation_timestamp + i

        start_warm_up_period = current_start_time_simulation_timestamp - warm_up_period_duration_seconds
        end_warm_up_period = current_start_time_simulation_timestamp

        current_attack_times = attack_times[
            (attack_times >= start_warm_up_period) & (attack_times < end_warm_up_period)
        ].copy()
        current_attack_times = current_attack_times - start_warm_up_period

        real_next_event_timestamp = attack_times[attack_times > end_warm_up_period][0] - start_warm_up_period

        sim_hawkes = get_hawkes_simulation(
            current_attack_times, warm_up_period_duration_seconds, prediction_period_duration_seconds, abb, baa, caa
        )

        predicted_timestamps = get_predicted_timestamps(sim_hawkes, warm_up_period_duration_seconds)
        predicted_next_event_timestamp = (
            predicted_timestamps[0]
            if len(predicted_timestamps) > 0
            else (end_warm_up_period - start_warm_up_period + prediction_period_duration_seconds)
        )

        prediction_error = predicted_next_event_timestamp - real_next_event_timestamp

        simulation_results_map['start_time_simulation_timestamp'].append(current_start_time_simulation_timestamp)
        simulation_results_map['real_next_event_timestamp'].append(start_warm_up_period + real_next_event_timestamp)
        simulation_results_map['predicted_next_event_timestamp'].append(start_warm_up_period + predicted_next_event_timestamp)
        simulation_results_map['prediction_error'].append(prediction_error)
        simulation_results_map['intensity'].append(sim_hawkes.tracked_intensity[0][-1])

    return pd.DataFrame(simulation_results_map)


def get_hawkes_simulation(
    attack_times: np.ndarray, warm_up_period_duration: float, prediction_period_duration: float,
    abb: float, baa: List[np.ndarray], caa:np.ndarray, seed: int = 1039
) -> hk.SimuHawkesExpKernels:
    sim_hawkes = hk.SimuHawkesExpKernels(
        adjacency=caa, decays=baa, baseline=[abb], end_time=warm_up_period_duration, seed=seed
    )
    sim_hawkes.track_intensity(1)

    sim_hawkes.set_timestamps([attack_times], warm_up_period_duration)
    t_max = warm_up_period_duration + prediction_period_duration
    sim_hawkes.end_time = t_max

    sim_hawkes.simulate()

    return sim_hawkes

def get_predicted_timestamps(sim_hawkes: hk.SimuHawkesExpKernels, warm_up_period_duration: float) -> np.ndarray:
    ay = sim_hawkes.timestamps
    ay_array = np.array(ay)
    condition = ay_array > warm_up_period_duration
    filtered_list = ay_array[condition]

    return filtered_list

def get_training_attack_times(attack_times: np.ndarray, start_time_training: float, end_time_training: float) -> np.ndarray:
    return attack_times[attack_times <= end_time_training] - start_time_training

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

def get_coe_testing_df_oracle(
    orderbook_df: pd.DataFrame,
    start_time_simulation_timestamp: float,
    simulation_end_time_timestamp_seconds: float,
    coe_training_start_time_timestamp_seconds: float
) -> pd.DataFrame:
    testing_df = orderbook_df[
        (orderbook_df['Timestamp'] >= start_time_simulation_timestamp * 1000) &
        (orderbook_df['Timestamp'] < simulation_end_time_timestamp_seconds * 1000)
    ][['Timestamp', 'BaseImbalance', 'Return']].copy()

    # keep only the first row for each second
    testing_df['SecondTimestamp'] = testing_df['Timestamp'] // 1000
    testing_df = testing_df.groupby('SecondTimestamp').first().reset_index()

    testing_df['Timestamp'] = (testing_df['Timestamp']/1000 - coe_training_start_time_timestamp_seconds)
    testing_df.drop(columns=['SecondTimestamp'], inplace=True)

    return testing_df

def get_coe_testing_df_hawkes(
    orderbook_df: pd.DataFrame, hawkes_result_df: pd.DataFrame, coe_training_start_time_timestamp_seconds: float
) -> pd.DataFrame:
    testing_df = pd.DataFrame({
        'TimestampPredicted' : hawkes_result_df['predicted_next_event_timestamp'],
        'TimestampOrderbook': (hawkes_result_df['real_next_event_timestamp'] * 1000).convert_dtypes(int),
    })
    testing_df = testing_df.merge(
        orderbook_df[['Timestamp', 'BaseImbalance', 'Return']],
        left_on='TimestampOrderbook',
        how='left',
        right_on='Timestamp'
    )
    testing_df = testing_df[['TimestampPredicted', 'BaseImbalance', 'Return']]
    testing_df.columns = ['Timestamp', 'BaseImbalance', 'Return']

    testing_df['Timestamp'] = (testing_df['Timestamp'] - coe_training_start_time_timestamp_seconds)

    return testing_df

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
        (orderbook_df['Timestamp'] <= simulation_end_time_timestamp_seconds * 1000)
    ].copy()
    testing_df = testing_df[['Timestamp', 'BaseImbalance', 'Return']]
    testing_df["DeltaT"] = testing_df["Timestamp"].shift(-1) - testing_df["Timestamp"]

    moving_average_window_size_milliseconds = moving_average_window_size_seconds * 1000
    moving_average_values = []
    for index, row in testing_df.iterrows():
        # Calculate the delta T using the rolling mean of the last 120 seconds
        ending_timestamp = row['Timestamp']
        starting_timestamp = ending_timestamp - moving_average_window_size_milliseconds
        moving_average = testing_df[
            (testing_df['Timestamp'] >= starting_timestamp) & (testing_df['Timestamp'] < ending_timestamp)
        ]['DeltaT'].mean() / 1000
        moving_average_values.append(moving_average)
    
    testing_df['DeltaT'] = moving_average_values
    testing_df = testing_df[testing_df['Timestamp'] >= start_time_simulation_timestamp * 1000]

    testing_df['Timestamp'] = (testing_df['Timestamp']/1000 - coe_training_start_time_timestamp_seconds)
    testing_df.drop(columns=['DeltaT'], inplace=True)

    return testing_df

def get_coe_testing_df_naive(
    orderbook_df: pd.DataFrame,
    coe_training_start_time_timestamp_seconds: float,
    start_time_simulation_timestamp: float,
    simulation_end_time_timestamp_seconds: float,
) -> pd.DataFrame:
    start_time_simulation_timestamp_milliseconds = start_time_simulation_timestamp * 1000
    simulation_end_time_timestamp_milliseconds = (simulation_end_time_timestamp_seconds + 1) * 1000
    rounded_df = orderbook_df[
        (orderbook_df['Timestamp'] >= start_time_simulation_timestamp_milliseconds) &
        (orderbook_df['Timestamp'] < simulation_end_time_timestamp_milliseconds)
    ].copy()
    rounded_df = rounded_df[['Timestamp', 'BaseImbalance', 'Return']]

    rounded_df['TimestampRounded'] = rounded_df['Timestamp'] // 1000
    rounded_df.drop_duplicates(subset=['TimestampRounded'], keep='last', inplace=True)
    rounded_df.drop(columns=['TimestampRounded'], inplace=True)

    testing_df_map = {
        'Timestamp': [],
        'BaseImbalance': [],
        'Return': [],
    }

    for timestamp in range(
        int(start_time_simulation_timestamp_milliseconds),
        int(simulation_end_time_timestamp_milliseconds),
        1000
    ):
        testing_df_map['Timestamp'].append(timestamp)
        next_base_imbalances = rounded_df[rounded_df['Timestamp'] >= timestamp]['BaseImbalance']
        if len(next_base_imbalances) == 0:
            testing_df_map['BaseImbalance'].append(testing_df_map['BaseImbalance'][-1])
        else:
            testing_df_map['BaseImbalance'].append(rounded_df[rounded_df['Timestamp'] >= timestamp]['BaseImbalance'].iloc[0])

        next_returns = rounded_df[rounded_df['Timestamp'] >= timestamp]['Return']
        if len(next_returns) == 0:
            testing_df_map['Return'].append(testing_df_map['Return'][-1])
        else:
            testing_df_map['Return'].append(rounded_df[rounded_df['Timestamp'] >= timestamp]['Return'].iloc[0])
    
    testing_df = pd.DataFrame(testing_df_map)
    testing_df['Timestamp'] = (testing_df['Timestamp']/ 1000) - coe_training_start_time_timestamp_seconds

    return testing_df


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

if __name__ == '__main__':
    simulation_period_duration_seconds = int(SIMULATION_DURATION.total_seconds())

    prediction_period_duration_seconds = PREDICTION_PERIOD_DURATION.total_seconds()
    warm_up_period_duration_seconds = WARM_UP_PERIOD_DURATION.total_seconds()

    coe_training_duration_seconds = COE_TRAINING_DURATION.total_seconds()

    with open(DENSITY_FILE_PATH) as json_file:
        file_densities_map = json.load(json_file)

    best_densities_file = pd.read_csv('/home/davide/Desktop/phd/hawkes/data/densities_table.csv')
    best_densities_timestamps = best_densities_file['timestamp'].values.tolist()
    best_densities_timestamps = [str(x) for x in best_densities_timestamps]
    best_densities_hours = best_densities_file['timestamp_density'].values.tolist()
    filename_intensities_map = dict()

    # getting already computed files
    already_computed_file_timestamps = set()
    for filename in os.listdir(HAWKES_DIRECTORY_PATH):
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

            start_time_training = max_density_start_time_simulation - TRAINING_DURATION
            start_time_training_timestamp = start_time_training.timestamp()

            end_time_simulation = max_density_start_time_simulation + SIMULATION_DURATION
            end_time_simulation_timestamp = end_time_simulation.timestamp()

            coe_training_start_time = max_density_start_time_simulation - COE_TRAINING_DURATION
            coe_training_start_time_timestamp = coe_training_start_time.timestamp()
            
            try:
                orderbook_df = pd.read_csv(filename_path, sep='\t')
                orderbook_df = get_preprocessed_df(orderbook_df, BASE_IMBALANCE_ORDERBOOK_LEVEL)
                attack_times = get_attack_times(orderbook_df['Timestamp'], start_time_training, end_time_simulation)

                training_attack_times = get_training_attack_times(
                    attack_times, start_time_training_timestamp, start_time_simulation_timestamp
                )
                best_decays = get_best_decays_with_cross_validation(training_attack_times)
                abb, baa, caa = get_hawkes_parameters_trained(training_attack_times, best_decays)

                hawkes_result_df = get_hawkes_results(
                    attack_times,
                    abb, baa, caa,
                    simulation_period_duration_seconds,
                    start_time_simulation_timestamp,
                    warm_up_period_duration_seconds,
                    prediction_period_duration_seconds,
                )

                intensities = hawkes_result_df['intensity'].values.tolist()
                hawkes_result_df.drop(columns=['intensity'], inplace=True)
                prediction_errors = hawkes_result_df['prediction_error'].values.tolist()
                hawkes_result_df.drop(columns=['prediction_error'], inplace=True)

                save_json_file(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    intensities,
                    'intensities'
                ) 

                save_json_file(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    prediction_errors,
                    'prediction_errors'
                ) 

                coe_training_df = get_coe_training_df(
                    orderbook_df, coe_training_start_time_timestamp, start_time_simulation_timestamp
                )
                hawkes_testing_coe_df = get_coe_testing_df_hawkes(
                    orderbook_df, hawkes_result_df, coe_training_start_time_timestamp
                )

                oracle_testing_coe_df = get_coe_testing_df_oracle(
                    orderbook_df, start_time_simulation_timestamp,
                    end_time_simulation_timestamp,
                    coe_training_start_time_timestamp
                )

                moving_average_testing_coe_df = get_coe_testing_df_moving_average(
                    orderbook_df,
                    coe_training_start_time_timestamp,
                    start_time_simulation_timestamp,
                    end_time_simulation_timestamp,
                    MOVING_AVERAGE_WINDOW_SIZE_SECONDS
                )

                naive_testing_coe_df = get_coe_testing_df_naive(
                    orderbook_df, coe_training_start_time_timestamp, start_time_simulation_timestamp, end_time_simulation_timestamp
                )

                save_complete_df(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    coe_training_df,
                    hawkes_testing_coe_df,
                    f"hawkes_BI{BASE_IMBALANCE_ORDERBOOK_LEVEL}"
                )
                save_complete_df(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    coe_training_df,
                    oracle_testing_coe_df,
                    f"oracle_BI{BASE_IMBALANCE_ORDERBOOK_LEVEL}"
                )
                save_complete_df(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    coe_training_df,
                    moving_average_testing_coe_df,
                    f"movingaverage_BI{BASE_IMBALANCE_ORDERBOOK_LEVEL}"
                )
                save_complete_df(
                    HAWKES_DIRECTORY_PATH,
                    filename_timestamp,
                    max_density_start_time_simulation_str,
                    coe_training_df,
                    naive_testing_coe_df,
                    f"naive_BI{BASE_IMBALANCE_ORDERBOOK_LEVEL}"
                )
                
            except Exception as e:
                print(f'Error while computing {filename_timestamp} with density {density_info[0]}: {e}')


