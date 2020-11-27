#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json


# Constants.
REMOVE_WHITESPACES = re.compile(r"\s+")

# Recieve enviroment variables.
DATASET_DIR = os.environ["DATASET_DIR"]
TRAINING_SETS = re.sub(REMOVE_WHITESPACES, "", os.environ["TRAIN_ON_SETS"]).split(",")
TARGET_SET = os.environ.get("TARGET_CURRENCY", 'BTC')
TRAINING_SETS.insert(0, TARGET_SET)


def parse_file(input_path: str, output_path: str) -> None:
    """Convert the given JSOn file to a CSV file.

    Args:
        input_path (str): Input JSON file path.
        output_path (str): Output CSV file path.
    """

    # Open files.
    input_file = open(input_path, "r")
    output_file = open(output_path, "w")

    # Write csv header.
    output_file.write("date,open,close,high,low,volume,vwap\n")

    try:
        # Access historical data.
        historical = json.load(input_file)["historical"]

        # Write historical data to the file.
        for i in historical:
            output_data = (
                f"{i['date']},{i['open']},{i['close']},{i['high']},{i['low']},{i['volume']},{i['vwap']}\n"
            )
            output_file.write(output_data)

    # Close all opend files.
    finally:
        input_file.close()
        output_file.close()


def main():
    """Loop though all enabled datasets and convert them to csv"""
    # Attempt creating directory.
    os.makedirs(f"{DATASET_DIR}/csv/", exist_ok=True)

    # Loop through all available datasets.
    for symbol in TRAINING_SETS:
        print(f"Processing {symbol}...")

        # Create paths and parse the data.
        input_path = f"{DATASET_DIR}/json/{symbol}.json"
        output_path = f"{DATASET_DIR}/csv/{symbol}.csv"
        parse_file(input_path, output_path)

    print("finished!")


if __name__ == "__main__":
    main()
