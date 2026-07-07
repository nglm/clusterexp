"""Helpers for discovering and converting datasets from the UCR archive."""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy.io import arff

from typing import List, Dict, Tuple, Union

from .data import process_labels

ILL_FORMATED = [
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "MelbournePedestrian", "AllGestureWiimoteX", "AllGestureWiimoteY",
    "AllGestureWiimoteZ", "GestureMidAirD1", "GestureMidAirD2",
    "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2",
    "PickupGestureWiimoteZ", "PLAID", "ShakeGestureWiimoteZ",
]
ILL_FORMATED_DIR = "Missing_value_and_variable_length_datasets_adjusted/"

# # Too many labels
# # (More than 20 in non-time series data, more than 15 in UCR)
# TOO_MANY_LABELS = [
#     # UCR
#     "PigArtPressure", "FiftyWords", "Adiac", "PigCVP", "Phoneme",
#     "PigAirwayPressure", "WordSynonyms", "NonInvasiveFetalECGThorax1",
#     "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
#     "Crop", "NonInvasiveFetalECGThorax2", "ShapesAll",
# ]

# # Too many samples
# # (More than 10000)
# TOO_MANY_SAMPLES = [
#     # UCR
#     "ElectricDevices", "Crop", "FordA", "FordB"
# ]

def find_datasets_UCR(
    path_ucr:str,
    with_ill_formated:bool = False,
) -> List[str]:
    """
    Find UCR datasets in a given folder.

    - recursively go through the given folder
    - finds files that ends with `_TRAIN.tsv`
    - extract the dataset name from the data file
    - returns a list of dataset names as ``[full/path/to/DATASET]``
      without the `_train.tsv`

    This will give the full path to the dataset, containing `path_ucr` as well

    Parameters
    ----------
    path_ucr : str
        Path to the folder containing datasets.
    with_ill_formated : bool, optional
        If ``True``, keep datasets from the adjusted directory that have
        missing values or variable lengths. If ``False``, exclude the
        ill-formatted variants.

    Returns
    -------
    List[str]
        List of dataset names found in the folder.
    """
    datasets = []
    for root, dirs, files in os.walk(path_ucr):
        data_files = [
            f.split('_TRAIN.tsv')[0] for f in files
            if f.endswith("_TRAIN.tsv")
        ]

        # Add full paths to the datasets list
        for name in data_files:
            datasets.append(os.path.join(root, name))

    if not with_ill_formated:
        # Filter out ill-formated datasets that are not in the adjusted folder
        datasets = [
            d for d in datasets
            if (
                any(ill in d for ill in ILL_FORMATED)
                and ILL_FORMATED_DIR in d
            ) or not any(ill in d for ill in ILL_FORMATED)
        ]

    # Remove common path_ucr from all dataset names
    common_path = os.path.commonpath(datasets)
    datasets = [d.replace(common_path, "") for d in datasets]

    return datasets

def save_data_labels_UCR(
    fnames: List[str],
    path_ucr: str = "",
    path_data: str = "",
) -> None:
    """
    Save UCR datasets to npy files.

    Parameters
    ----------
    fnames : List[str]
        Dataset base names relative to ``path_ucr`` without the
        ``_TRAIN.tsv`` suffix.
    path_ucr : str
        Path to the UCR data folder.
    path_data : str, optional
        Destination directory where ``*_data.npy`` and
        ``*_labels.npy`` files will be written.

    Returns
    -------
    None
        This function writes NumPy arrays to disk and returns nothing.
    """

    # Make sure all path exists otherwise create it
    for fname in fnames:

        # Get data and labels from the original UCR dataset folder
        data, labels = get_data_labels_UCR(f"{path_ucr}{fname}_TRAIN.tsv")

        # Prepare destination
        p = Path(f"{path_data}{fname}_data.npy")
        p.parent.mkdir(parents=True, exist_ok=True)

        # Save labels and data to npy files in the destination folder
        np.save(f"{path_data}{fname}_labels.npy", labels)
        np.save(f"{path_data}{fname}_data.npy", data)

def get_data_labels_UCR(
    fname: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a UCR training split and return reshaped data with labels.

    Parameters
    ----------
    fname : str
        Path to the UCR TSV file (including `_TRAIN.tsv`).


    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Time-series data reshaped as ``(N, T, 1)`` together with the
        encoded labels.
    """

    df = pd.read_csv(f"{fname}", sep="\t")

    data = np.expand_dims(df.iloc[:, 1:].to_numpy(), axis=2)

    labels = df.iloc[:, 0].to_numpy()
    labels, n_labels = process_labels(labels)

    return data, labels