#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import requests

import numpy as np
from settings import *


def download_currency(symbol: str) -> None:
    """Download a dataset of the given symbol.

    Args:
        symbol string: Currency that should be requested. For example BTC / ETH.

    Raises:
        Exception: When writing the file or making the request failed.
    """
    with open(f"{DATASET_DIR}/json/{symbol}.json", "wb") as file:

        # Craft the url where the dataset could be downloaded from and make a get request.
        url = API_URL.format(
            symbol=symbol,
            comp_sym=COMPARING_CURRENCY,
            key=KEY,
            data="historical-price-full",
        )
        result = requests.get(url)

        # Handle data recieval exception.
        if result.status_code != 200:
            raise Exception(
                f"Request failed error {result.status_code}\n{result.content}"
            )

        # Write the content to a file
        file.write(result.content)


def download_everything(overwrite=False):
    """Download all not downloaded datasets.
    """

    # Create a directory where all files should be stored in
    os.makedirs(f"{DATASET_DIR}/json/", exist_ok=True)

    # Loop through all datasets it should download.
    for symbol in TRAINING_SETS:

        # Ignore datasets that are already downloaded.
        if os.path.isfile(f"{DATASET_DIR}/json/{symbol}.json") and not overwrite:
            print(f"{symbol} already exists. skipping")
            continue

        # Download the dataset.
        print(f"Downloading {symbol}...")
        download_currency(symbol)

        # Wait a couple of seconds to prevent throttling.
        time.sleep(5)


def recieve_update(symbol: str, data: str = "4hour") -> list:
    """Recieve new data from the financial api.

    Args:
        symbol (str): Symbol of the currency for example BTC
        data (str, optional): Range of the date. Defaults to '4hour'.

    Returns:
        np.array: row of data from the given symbol
    """

    # Call data from the api
    url = API_URL.format(
        symbol=symbol,
        comp_sym=COMPARING_CURRENCY,
        key=KEY,
        data="historical-chart/" + data,
    )
    result = requests.get(url).content
    raw_data = json.loads(result)[0]

    # Create a row of features
    output = []
    for feature in FEATURES:
        output.append(raw_data[feature])

    return output


def recieve_updates(data: str = "4hour") -> np.array:
    output = []
    for symbol in TRAINING_SETS:
        output += recieve_update(symbol, data)

    return output


if __name__ == "__main__":
    download_everything()
