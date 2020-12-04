import time
import random
from . import scaler
from .train import Forcaster
from . import data_handler as dh

from settings import *


def load_model():

    # Load the scalar
    scaler.load_scaler(f"{MODEL_DIR}/{MODEL_NAME}.pickle")

    # Load a temponary dataset to recieve all the settings
    x_set, y_set = dh.load_datasets(after_date=START_DATE)
    x_set, y_set = dh.create_window(x_set, y_set, window_size=TRAINING_WINDOW)

    # Create the model based on these settings
    model = Forcaster(x_set[0].shape, y_set[0].shape[0])

    model.load_weights(f"{MODEL_DIR}/{MODEL_NAME}.h5")

    return model


if __name__ == "__main__":
    model = load_model()
    window = dh.latest_window()

    predicted = model.predict(window)

    print("Predicted price for the next day:", scaler.unscale_y(predicted)[0])
