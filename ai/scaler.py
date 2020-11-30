import pickle
from sklearn.preprocessing import MinMaxScaler

# Dirty import
from settings import *

# Global variable
SCALERS = None


def min_max_scaler(train_x: np.array, train_y: np.array) -> (np.array, np.array):
    """Apply the min-max scaler on the given data

    Args:
        train_x (np.array): Training data
        train_y (np.array): Labeling data

    Returns:
        (np.array, np.array): (train_x, train_y) with min-max applied
    """
    global SCALERS
    if SCALERS:
        train_x = SCALERS[0].transform(train_x)
        train_y = SCALERS[1].transform(train_y)

    # Create the scaler if it doesnt exist yet.
    else:
        SCALERS = [MinMaxScaler(), MinMaxScaler()]
        train_x = SCALERS[0].fit_transform(train_x)
        train_y = SCALERS[1].fit_transform(train_y)

    return train_x, train_y


def unscale(train_x: np.array, train_y: np.array) -> (np.array, np.array):
    """Undo the set scaler and transform the data back in normal data

    Args:
        train_x (np.array): Should be the same shape as before getting scaled.
        train_y (np.array): -

    Returns:
        (np.array, np.array): The given data transformed in it's old glory.
    """
    global SCALERS
    train_x = SCALERS[0].inverse_transform(train_x)
    train_y = SCALERS[1].inverse_transform(train_y)
    return train_x, train_y


def scale_x(train_x: np.array) -> np.array:
    return SCALERS[0].transform(train_x)


def unscale_x(train_x: np.array, window: bool = False) -> np.array:
    """Transform feature data back into it's old glory

    Args:
        train_x (np.array): Should be the same shape as before getting scaled.
        window (bool, optional): Enable when windowing has been applied to the data. Defaults to False.

    Returns:
        np.array: Data transformed back
    """
    global SCALERS

    if window:
        train_x = train_x.reshape(-1, train_x.shape[-1])
    return SCALERS[0].inverse_transform(train_x)


def unscale_y(train_y: np.array, window: bool = False) -> np.array:
    """Transform feature data back into it's old glory

    Args:
        train_y (np.array): Should be the same shape as before getting scaled.
        window (bool, optional): Enable when windowing has been applied to the data. Defaults to False.

    Returns:
        np.array: Data transformed back
    """
    global SCALERS

    if window:
        train_y = [train_y]
    return SCALERS[1].inverse_transform(train_y)


def store_scaler(path: str) -> None:
    """Store the scaler settings to a file

    Args:
        path (str): path to the file it should be stored in.
    """
    with open(path, "wb") as file:
        pickle.dump(SCALERS, file)


def load_scaler(path: str) -> None:
    """Load the scaler settings from a file into memory

    Args:
        path (string): [description]
    """
    global SCALERS
    with open(path, "rb") as file:
        SCALERS = pickle.load(file)
