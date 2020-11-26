#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to access and convert the dataset into useable data.

Author: Vepnar (Arjan de Haan)
"""

import os
import re
import numpy as np
import pandas as pd

# Constants.
REMOVE_WHITESPACES = re.compile(r"\s+")
DTYPE = {"open": np.float32, "close": np.float32, "high": np.float32, "low": np.float32}

# Recieve enviroment variables.
DATASET_DIR = os.environ["DATASET_DIR"]
TRAINING_SETS = re.sub(REMOVE_WHITESPACES, "", os.environ["TRAIN_ON_SETS"]).split(",")


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
    """Count the amount of rows in a dataset

    Args:
        after_date (str):  Only use data points after this date.
    """
    for symbol in TRAINING_SETS:
        print(symbol, len(load_dataset(symbol, after_date)))


def load_datasets(
    target: str, after_date: str, remove_date: bool = True
) -> (np.array, np.array):
    # TODO add comments
    train_y = []
    train_x = pd.DataFrame()
    for symbol in TRAINING_SETS:
        raw_df = load_dataset(symbol, after_date)

        # Merge dataframes when this is possible
        if train_x.empty:
            train_x = raw_df
        else:
            train_x = train_x.merge(raw_df, how="outer", on="date")

        # Set the open price as the Y data when the symbols match
        if symbol == target:
            train_y = raw_df["open"]

    # Drop rows with missing data.
    train_x.dropna(inplace=True)

    if remove_date:
        train_x.drop("date", 1, inplace=True)

    return train_x.to_numpy(), train_y.to_numpy()


def create_window(self, train_x: np.array, train_y: np.array, window_size: int = 30):
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
    """
    windows = len(train_x) - window_size - 1
    window_x = []
    window_y = []
    for i in range(0, windows):
        window_x.append(train_x[i:window_size])
        window_y.append(train_y[i + window_size])
