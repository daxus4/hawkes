import os
from typing import Dict, List

import pandas as pd
import yaml

from constants import (
    COE_TRAINING_DURATION_MINUTES,
    CONF_SAMPLING_FILE_PATH,
    DATA_DIRECTORY,
    SAMPLED_COE_DF_SUBDIRECTORY,
)


def get_test_df(complete_df: pd.DataFrame, testing_start_time: float) -> pd.DataFrame:
    return complete_df[complete_df["Timestamp"] >= testing_start_time].copy()


def get_training_df(
    complete_df: pd.DataFrame, training_end_time: float
) -> pd.DataFrame:
    return complete_df[complete_df["Timestamp"] < training_end_time].copy()


def _get_index_positions_without_old_greater_predictions(
    series: pd.Series,
) -> pd.Series:
    to_delete_index_positions = []
    for i in range(0, len(series)):
        for j in range(i + 1, len(series)):
            if series.iloc[i] > series.iloc[j]:
                to_delete_index_positions.append(i)
                break

    return to_delete_index_positions


def _get_dataframe_without_index_positions(
    df: pd.DataFrame, index_positions: List
) -> pd.DataFrame:
    return df.drop(df.index[index_positions])


def get_dataframe_without_old_greater_predictions(
    df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    series = df[column_name]
    to_delete_index_positions = _get_index_positions_without_old_greater_predictions(
        series
    )
    df = _get_dataframe_without_index_positions(df, to_delete_index_positions)

    return df


def get_method_type_dataframe_map(
    data_path: str, submethods_folders: List[str], filename: str
) -> Dict[str, Dict[str, pd.DataFrame]]:
    return {
        submethod_folder: get_type_dataframe_map(
            pd.read_csv(os.path.join(data_path, submethod_folder, filename))
        )
        for submethod_folder in submethods_folders
    }


def get_type_dataframe_map(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    training_end_time = COE_TRAINING_DURATION_MINUTES * 60
    return {
        "train": get_training_df(df, training_end_time),
        "test": get_dataframe_without_old_greater_predictions(
            get_test_df(df, training_end_time), "Timestamp"
        ),
    }


def get_minimun_index_length_for_test_dfs(
    method_type_dataframe_map: Dict[str, Dict[str, pd.DataFrame]]
) -> int:
    return get_minimun_index_length(
        [
            method_dataframe_map["test"]
            for method_dataframe_map in method_type_dataframe_map.values()
        ]
    )


def get_minimun_index_length(dfs: List[pd.DataFrame]) -> int:
    return min([len(df.index) for df in dfs])


def get_method_dataframe_map_for_coe(
    method_type_dataframe_map: Dict[str, Dict[str, pd.DataFrame]]
) -> Dict[str, pd.DataFrame]:
    minimun_index_length = get_minimun_index_length_for_test_dfs(
        method_type_dataframe_map
    )

    return {
        method: pd.concat(
            [
                method_dataframe_map["train"],
                get_sampled_sorted_df(
                    method_dataframe_map["test"], "Timestamp", minimun_index_length
                ),
            ]
        )
        for method, method_dataframe_map in method_type_dataframe_map.items()
    }


def get_sampled_sorted_df(
    df: pd.DataFrame, column_name: str, sample_number: int
) -> pd.DataFrame:
    df = df.sample(sample_number)
    df = df.sort_values(by=column_name)

    return df


def compute_sampled_dataframes_for_coe(
    data_path: str,
    submethods_folders: List[str],
    results_dir: str,
    base_imbalance_level: int,
) -> None:
    filenames = os.listdir(os.path.join(data_path, submethods_folders[0]))
    filenames = [
        filename
        for filename in filenames
        if filename.endswith(f"_BI{base_imbalance_level}.csv")
    ]

    for filename in filenames:

        method_type_dataframe_map = get_method_type_dataframe_map(
            data_path, submethods_folders, filename
        )
        method_dataframe_map = get_method_dataframe_map_for_coe(
            method_type_dataframe_map
        )

        for method, df in method_dataframe_map.items():
            df.to_csv(os.path.join(results_dir, method, filename), index=False)


def get_conf(path: str):
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return conf


def main() -> None:
    results_path = os.path.join(DATA_DIRECTORY, SAMPLED_COE_DF_SUBDIRECTORY)

    conf = get_conf(CONF_SAMPLING_FILE_PATH)
    submethods_folders = conf["folders"]
    base_imbalance_level = conf["base_imbalance_level"]

    _create_results_dirs(submethods_folders, results_path)

    compute_sampled_dataframes_for_coe(
        DATA_DIRECTORY, submethods_folders, results_path, base_imbalance_level
    )


def _create_results_dirs(submethods_folders, results_path):
    os.makedirs(results_path, exist_ok=True)

    for submethod_folder in submethods_folders:
        os.makedirs(os.path.join(results_path, submethod_folder), exist_ok=True)


if __name__ == "__main__":
    main()
