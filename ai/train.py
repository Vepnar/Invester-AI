import os
from . import data_handler as dh
from tensorflow.keras import Sequential, layers

from settings import *


def Forcaster(input_shape, output_shape, units=32):
    return Sequential(
        [
            layers.LSTM(units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.1),
            layers.LSTM(units),
            layers.Dropout(0.1),
            layers.Dense(output_shape),
        ]
    )


def main():
    # Load modles from the file
    raw_x, raw_y = dh.load_datasets()

    train_x = raw_x[TEST_SIZE:]
    train_y = raw_y[TEST_SIZE:]

    test_x = raw_x[:TEST_SIZE]
    test_y = raw_y[:TEST_SIZE]

    # Scale train & test data
    train_x, train_y = dh.min_max_scaler(train_x, train_y)
    test_x, test_y = dh.min_max_scaler(test_x, test_y)

    # Create training windows
    train_x, train_y = dh.create_window(train_x, train_y, window_size=TRAINING_WINDOW)
    test_x, test_y = dh.create_window(test_x, test_y, window_size=TRAINING_WINDOW)

    # Create machine learning model
    model = Forcaster(train_x[0].shape, train_y[0].shape[0])
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    print("Begin training...")
    model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )

    # Display the current model
    print(model.summary())

    # Save the model to a file
    print(f"Saving model to {MODEL_DIR}/{MODEL_NAME}")
    model.save(f"{MODEL_DIR}/{MODEL_NAME}.h5")
    dh.store_scaler(f"{MODEL_DIR}/{MODEL_NAME}.pickle")


if __name__ == "__main__":

    # Make dirs that don't exist yet
    os.makedirs(MODEL_DIR, exist_ok=True)

    main()
