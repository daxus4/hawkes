import os
from typing import Dict, List, Tuple

import pandas as pd
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from constants import DATA_DIRECTORY


def get_train_test_data(
    coe_df: pd.DataFrame, max_training_time_seconds: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    return (
        coe_df[coe_df["Timestamp"] <= max_training_time_seconds],
        coe_df[coe_df["Timestamp"] > max_training_time_seconds],
    )


def get_feature_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    return df[["BaseImbalance"]], df["Return"]


def get_fitted_linear_regression(
    train_features: pd.DataFrame, train_target: pd.Series
) -> LinearRegression:
    model = LinearRegression()
    model.fit(train_features, train_target)

    return model


def get_mse_bi_prediction(
    coe_df: pd.DataFrame, max_training_time_seconds: float
) -> float:
    train_df, test_df = get_train_test_data(coe_df, max_training_time_seconds)
    train_features, train_target = get_feature_and_target(train_df)
    test_features, test_target = get_feature_and_target(test_df)

    model = get_fitted_linear_regression(train_features, train_target)
    prediction = model.predict(test_features)

    return mean_squared_error(test_target, prediction)


def get_mses_bi_predictions_for_submethod(
    submethod_dir: str, max_training_time_seconds: float
) -> List[float]:
    mses = list()
    for filename in os.listdir(submethod_dir):
        if filename.endswith(".csv"):
            coe_df = pd.read_csv(os.path.join(submethod_dir, filename))
            mses.append(get_mse_bi_prediction(coe_df, max_training_time_seconds))

    return mses


def get_mses_bi_predictions_for_selected_submethods(
    data_path: str, selected_submethods: List[str], max_training_time_seconds: float
) -> Dict[str, List[str]]:

    return {
        submethod_folder: get_mses_bi_predictions_for_submethod(
            os.path.join(data_path, submethod_folder), max_training_time_seconds
        )
        for submethod_folder in os.listdir(data_path)
        if submethod_folder in selected_submethods
    }


def get_mses_dataframe(
    data_path: str, selected_submethods: List[str], max_training_time_seconds: float
) -> pd.DataFrame:
    mses = get_mses_bi_predictions_for_selected_submethods(
        data_path, selected_submethods, max_training_time_seconds
    )

    df_map = {"submethod": list(), "mse": list()}
    for submethod, mse_list in mses.items():
        for mse in mse_list:
            df_map["submethod"].append(submethod)
            df_map["mse"].append(mse)

    return pd.DataFrame(df_map)


if __name__ == "__main__":
    selected_submethods = os.listdir(os.path.join(DATA_DIRECTORY, "sampled_coe_dfs"))
    max_training_time_seconds = 1000

    df = get_mses_dataframe(
        os.path.join(DATA_DIRECTORY, "sampled_coe_dfs"),
        selected_submethods,
        max_training_time_seconds,
    )

    df.to_csv(os.path.join(DATA_DIRECTORY, "mses.tsv"), index=False, sep="\t")
