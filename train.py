
from tensorflow.keras import Sequential, layers
import dataset as dt

def Forcaster(input_shape, output_shape, units=32):
    return Sequential([
        layers.LSTM(units, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.1),
        layers.LSTM(units),
        layers.Dropout(0.1),
        layers.Dense(output_shape)
    ])


def main(epoch=200, batch_size=100, test_size=140, training_window=30, output_file="model.h5"):

    # Load modles from the file
    raw_x, raw_y = dt.load_datasets()

    train_x = raw_x[test_size:]
    train_y = raw_y[test_size:]

    test_x = raw_x[:test_size]
    test_y = raw_y[:test_size]

    # Scale train & test data
    train_x, train_y = dt.min_max_scaler(train_x, train_y)
    test_x, test_y = dt.min_max_scaler(test_x, test_y)

    # Create training windows
    train_x, train_y = dt.create_window(train_x, train_y, window_size=training_window)
    test_x, test_y = dt.create_window(test_x, test_y, window_size=training_window)

    # Create machine learning model
    model = Forcaster(train_x[0].shape, train_y[0].shape[0])
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    print("Begin training...")
    model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        batch_size=batch_size,
        epochs=epoch,
    )

    # Display the current model
    print(model.summary())

    # Save the model to a file
    print(f"Saving model to {output_file}")
    model.save(output_file)

if __name__ == "__main__":
    main()