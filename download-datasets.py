#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import typing
import time
import requests

# Constants
REMOVE_WHITESPACES = re.compile(r"\s+")
URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}{comp_sym}?apikey={key}"

# Enviroment variables
KEY = os.environ["KEY"]
TRAINING_SETS = re.sub(REMOVE_WHITESPACES, "", os.environ["TRAIN_ON_SETS"]).split(",")
COMPARING_CURRENCY = os.environ["COMPARING_CURRENCY"]
DATASET_DIR = os.environ["DATASET_DIR"]


def download_currency(symbol: str) -> None:
    """Download a dataset of the given symbol.

    Args:
        symbol string: Currency that should be requested. For example BTC / ETH.

    Raises:
        Exception: When writing the file or making the request failed.
    """
    with open(f"{DATASET_DIR}/json/{symbol}.json", "wb") as file:

        # Craft the url where the dataset could be downloaded from and make a get request.
        url = URL.format(symbol=symbol, comp_sym=COMPARING_CURRENCY, key=KEY)
        result = requests.get(url)

        # Handle data recieval exception.
        if result.status_code != 200:
            raise Exception(
                f"Request failed error {result.status_code}\n{result.content}"
            )

        # Write the content to a file
        file.write(result.content)


def download_everything():
    """Download all not downloaded datasets.
    """

    # Create a directory where all files should be stored in
    os.makedirs(f"{DATASET_DIR}/json/", exist_ok=True)

    # Loop through all datasets it should download.
    for symbol in TRAINING_SETS:

        # Ignore datasets that are already downloaded.
        if os.path.isfile(f"{DATASET_DIR}/json/{symbol}.json"):
            print(f"{symbol} already exists. skipping")
            continue

        # Download the dataset.
        print(f"Downloading {symbol}...")
        download_currency(symbol)

        # Wait a couple of seconds to prevent throttling.
        time.sleep(5)


if __name__ == "__main__":
    download_everything()
