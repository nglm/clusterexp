import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.io import arff

from typing import List, Dict, Tuple, Union

from .utils import process_labels

HOME_DIR = os.path.expanduser('~')
PATH_UCR_LOCAL = f"{HOME_DIR}/Documents/Work/Data/UCR/UCRArchive_2018/"
PATH_UCR_REMOTE = f"{HOME_DIR}/UCR/UCRArchive_2018/"

ILL_FORMATED = [
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "MelbournePedestrian", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "GestureMidAirD1", "GestureMidAirD2",
    "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2",
    "PickupGestureWiimoteZ", "PLAID", "ShakeGestureWiimoteZ",
    ]
ILL_FORMATED_DIR = "Missing_value_and_variable_length_datasets_adjusted/"

# Too many labels
# (More than 20 in non-time series data, more than 15 in UCR)
TOO_MANY_LABELS = [
    # UCR
    "PigArtPressure", "FiftyWords", "Adiac", "PigCVP", "Phoneme",
    "PigAirwayPressure", "WordSynonyms", "NonInvasiveFetalECGThorax1",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "Crop", "NonInvasiveFetalECGThorax2", "ShapesAll",
]

# Too many samples
# (More than 10000)
TOO_MANY_SAMPLES = [
    # UCR
    "ElectricDevices", "Crop", "FordA", "FordB"
]


def get_data_labels_UCR(
    fname: str,
) -> Tuple[np.ndarray, Union[None, np.ndarray], int]:
    """
    Get dataset, labels number of labels for UCR data

    Parameters
    ----------
    fname : str
        Path to the UCR TSV file.

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], int, None]
        Time-series data reshaped as ``(N, T, 1)``, encoded labels,
        number of labels.
    """

    df = pd.read_csv(fname, sep="\t")

    data = np.expand_dims(df.iloc[:, 1:].to_numpy(), axis=2)

    labels = df.iloc[:, 0].to_numpy()
    labels, n_labels = process_labels(labels)

    return data, labels