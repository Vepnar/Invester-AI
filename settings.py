import os
import re
import numpy as np
from datetime import datetime


# Don't change
env = os.environ.get
REMOVE_WHITESPACES = re.compile(r"\s+")
DATE_PARSER = lambda date: datetime.strptime(date, "%Y-%m-%d")
DTYPE = {"open": np.float64, "close": np.float64}

# API settings
KEY = os.environ["KEY"]
API_URL = (
    "https://financialmodelingprep.com/api/v3/{data}/{symbol}{comp_sym}?apikey={key}"
)

# Data settings
FEATURES = re.sub(REMOVE_WHITESPACES, "", env("FEATURES", "open,close")).split(",")
TRAINING_SETS = re.sub(REMOVE_WHITESPACES, "", env("TRAIN_ON_SETS", "")).split(",")
COMPARING_CURRENCY = env("COMPARING_CURRENCY", "USD")
DATASET_DIR = env("DATASET_DIR", "./dataset")
TARGET_SET = env("TARGET_CURRENCY", "BTC")
TRAINING_SETS.insert(0, TARGET_SET)


# Training settings
MODEL_DIR = env("MODEL_DIR", "./model")
MODEL_NAME = env("MODEL_NAME", "Jan")
EPOCH = int(env("EPOCH", 200))
BATCH_SIZE = int(env("BATCH_SIZE", 100))
TEST_SIZE = int(env("TEST_SIZE", 100))
TRAINING_WINDOW = int(env("TRAINING_WINDOW", "10"))
START_DATE = env("START_DATE", "2019-1-1")
