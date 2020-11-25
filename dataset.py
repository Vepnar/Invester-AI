#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from datetime import date

# Constants.
REMOVE_WHITESPACES = re.compile(r"\s+")
DTYPE = {'open': np.float32, 'close': np.float32,  'high': np.float32,  'low': np.float32}

# Recieve enviroment variables.
DATASET_DIR = os.environ["DATASET_DIR"]
TRAINING_SETS = re.sub(REMOVE_WHITESPACES, "",
                       os.environ["TRAIN_ON_SETS"]).split(",")


def load_dataset(symbol: str, after_date: str, remove_date : bool=False) -> pd.DataFrame:
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
        df.drop(['date'], 1, inplace=True)
    return df


def count_datapoints(after_date: str) -> None:
    """Count the amount of rows in a dataset

    Args:
        after_date (str):  Only use data points after this date.
    """
    for symbol in TRAINING_SETS:
        print(symbol, len(load_dataset(symbol, after_date)))


def load_datasets(target: str, after_date: str, remove_date:bool=True) -> (np.array, np.array):
    # TODO add comments
    y = []
    x_df = pd.DataFrame()
    for symbol in TRAINING_SETS:
        print(symbol)
        df = load_dataset(symbol, after_date)

        # Merge dataframes when this is possible
        if x_df.empty:
            x_df = df
        else: 
            x_df= x_df.merge(df, how='outer', on='date')

        # Set the open price as the Y data when the symbols match
        if symbol == target:
            y = df['open']

    # Drop rows with missing data
    x_df.dropna(inplace=True)

    if remove_date:
        x_df.drop('date', 1, inplace=True)

    return x_df.to_numpy(), y.to_numpy()
