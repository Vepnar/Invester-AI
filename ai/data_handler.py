#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to access and convert the dataset into useable data.

Author: Vepnar (Arjan de Haan)
"""

import os
import re
import pandas as pd
from datetime import datetime
from . import scaler

# Dirty import
from settings import *


def parse_dataframe(symbol: str, **kwargs) -> pd.DataFrame:
    """Load a CSV file from the given cryptocurrency

    Args:
        symbol (str): For example BTC

    Returns:
        pd.DataFrame: A dataframe with all the rules applied.
    """
    return pd.read_csv(
        f"{DATASET_DIR}/csv/{symbol}.csv",
        dtype=DTYPE,
        parse_dates=["date"],
        date_parser=DATE_PARSER,
        **kwargs,
    )


def load_dataset(
    symbol: str, after_date: str, remove_date: bool = False
) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Args:
        symbol (str): Type of stock / currency you want to load.
        after_date (str): Only use data points after this date.
        remove_date (bool, optional): Remove the date row. Defaults to False.

    Returns:
        pd.DataFrame: An dataframe with all usefull data
    """
    df = pd.read_csv(f"{DATASET_DIR}/csv/{symbol}.csv", dtype=DTYPE, parse_dates=True)
    df.sort_values(by=["date"], inplace=True)
    df = df[(after_date < df["date"])]
    if remove_date:
        df.drop(["date"], 1, inplace=True)
    return df


def count_datapoints(after_date: str) -> None:
    """Print the amount of rows in a dataset

    Args:
        after_date (str):  Only use data points after this date.
    """
    for symbol in TRAINING_SETS:
        print(symbol, len(load_dataset(symbol, after_date)))


def load_datasets(
    after_date: str = "2015-1-1", remove_date: bool = True
) -> (np.array, np.array):
    """Load all enabled datasets and turn them into train and label data.

    Args:
        after_date (str, optional): Only use data after this date. Defaults to '2015-1-1'
        remove_date (bool, optional): Remove the date column if this is enabled. Defaults to True

    Returns:
        (np.array, np.array): train_x & train_y
    """
    train_x, train_y = pd.DataFrame(), []

    # Iterate through all enabled datasets
    for symbol in TRAINING_SETS:
        raw_df = load_dataset(symbol, after_date)

        # Merge dataframes when this is possible
        if train_x.empty:
            train_x = raw_df
        else:
            train_x = train_x.merge(raw_df, how="outer", on="date")

        # Set the open price as the Y data when the symbols match
        if symbol == TARGET_SET:
            train_y = raw_df[["open", "close"]]

    # Drop rows with missing data.
    train_x.dropna(inplace=True)

    # Drop the date column
    if remove_date:
        train_x.drop("date", 1, inplace=True)

    return train_x.to_numpy(), train_y.to_numpy()


def create_window(
    train_x: np.array, train_y: np.array, window_size: int = 30
) -> (np.array, np.array):
    """Convert the given data set into a sequence window.

    Turn: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Into:
        - [1, 2, 3] [4]
        - [2, 3, 4] [5]
        - [3, 4, 5] [6]

    Except with given x and y values.


    Args:
        train_x (np.array): X array. Allowed to be multi dimensional.
        train_y (np.array): Y array. Not allowed to be multi dimensional.
        window_size (int, optional): The size of the sequence window. Defaults to 30.
    
    returns:
        (np.array, np.array): X window & Y window.
    """
    windows = len(train_x) - window_size - 1
    window_x = []
    window_y = []
    for i in range(0, windows):
        window_x.append(train_x[i : i + window_size])
        window_y.append(train_y[i + window_size])

    return np.asarray(window_x), np.asarray(window_y)


def dataset_age() -> int:
    """Count the age of the oldest dataset.

    Returns:
        int: Age of the oldest dataset
    """
    age, now = 0, datetime.now()

    # Loop though all active datasets.
    for symbol in TRAINING_SETS:

        # Select the newest row and compute the difference.
        raw_df = parse_dataframe(symbol, nrows=2)
        last = raw_df.iloc[0]["date"]
        difference = (now - last).days

        # Update the dage when it's older than the oldest dataset.
        if difference > age:
            age = difference

    return age


def latest_window():
    # TODO add documentation
    # Loop though all active datasets.
    window = pd.DataFrame()

    for symbol in TRAINING_SETS:

        raw_df = parse_dataframe(symbol, nrows=TRAINING_WINDOW)

        # Merge dataframes when this is possible
        if window.empty:
            window = raw_df
        else:
            window = window.merge(raw_df, how="outer", on="date")

    # remove all the dates
    window.drop(["date"], 1, inplace=True)

    # Scale the window and turn it into a numpy array.
    return np.asarray([scaler.scale_x(window.to_numpy())])
