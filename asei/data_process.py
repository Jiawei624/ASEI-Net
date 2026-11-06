"""Convert long-format CSV records into time-series arrays and data frames."""
from typing import Union

import numpy as np
import pandas as pd


def is_st(data: pd.DataFrame) -> bool:
    """
    Checks if data of the CSV file are in spatial-temporal format.

    :param data: The series data.
    :return: Are all values in 'cols' column are in spatial-temporal format.
    """
    return {"date", "id", "cols", "data"}.issubset(data.columns)


def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Read the data file and return DataFrame.If the data is spatial-temporal format,

    return it as a numpy array; otherwise, return it as a Pandas DataFrame.

    :param path: The path to the data file.
    :return:  The content of the data file.
    """
    data = pd.read_csv(path)
    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.

    According to the provided file path, read the data file and return the corresponding DataFrame.

    :param data: Data frame to read.
    :return:  The DataFrame of the content of the data file.
    """
    if data.empty or "cols" not in data or data["cols"].isna().any():
        raise ValueError("Long-format data requires nonempty, named channels.")
    value_columns = [name for name in data if name not in {"cols", "date"}]
    if len(value_columns) != 1:
        raise ValueError("Long-format data requires one value column, cols and optional date.")
    value = value_columns[0]
    names = data["cols"].unique().tolist()
    # TAB files have millions of rows in contiguous channel blocks. Avoid a
    # full-table pivot/copy while still accepting interleaved long-format rows.
    channel_values = data["cols"].to_numpy()
    edges = np.r_[0, np.flatnonzero(channel_values[1:] != channel_values[:-1]) + 1, len(data)]
    if len(edges) - 1 == len(names):
        groups = ((channel_values[left], data.iloc[left:right]) for left, right in zip(edges[:-1], edges[1:]))
    else:
        groups = data.groupby("cols", sort=False)
    if "date" in data:
        timeline, columns = None, {}
        for name, group in groups:
            times = pd.DatetimeIndex(pd.to_datetime(group["date"], errors="raise"))
            if times.hasnans or not times.is_unique:
                raise ValueError("Duplicate or missing channel timestamps in long-format data.")
            if timeline is None:
                timeline = times.sort_values()
            order = times.get_indexer(timeline)
            if len(times) != len(timeline) or (order < 0).any():
                raise ValueError("Long-format channels must share the same timestamp grid.")
            columns[name] = group[value].to_numpy()[order]
        df = pd.DataFrame(columns, index=timeline)
        df.index.name = "date"
    else:
        columns = {name: group[value].to_numpy() for name, group in groups}
        if len({len(group) for group in columns.values()}) != 1:
            raise ValueError("Long-format channels have unequal lengths.")
        df = pd.DataFrame(columns)
    df.columns.name = None

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    """
    Convert spatial-temporal data from a DataFrame

    to a three-dimensional(time stamp,feature,sensor)  numpy array.

    :param df: Spatial-temporal data.
    :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
    :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
    """
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

    sensors = df["id"].unique()
    features = df["cols"].unique()
    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )

    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
    data_np = np.transpose(data_np, (0, 2, 1))

    if nrows is not None:
        data_np = data_np[:nrows, :, :]

    return data_np
