"""Helpers for fetching and converting Barton benchmark datasets."""

import io
import urllib.request
from scipy.io import arff

import pandas as pd
import numpy as np
import os

from typing import List, Dict, Tuple, Union

from .data import process_labels

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'

# Just one cluster
UNIMODAL = [
    "birch-rg1.arff", "birch-rg2.arff",
    "golfball.arff",
]
# No class column in the data
UNLABELED = [
    # artificial
    "birch-rg1.arff", "birch-rg2.arff",
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",

    # real-world
    "water-treatment.arff",
]

# Unknown number of clusters
UNKNOWN_K = [
    # artificial
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",

    # real-world
]

# Datasets removed, for various reasons (e.g. missing data)
INVALID = [
    #"segment.arff",
    # Contains missing values
    'dermatology.arff',
    "water-treatment.arff",
    # give arff error: "String attributes not supported yet, sorry"
    "yeast.arff",
]

# N_SAMPLES_MAX = 10000

# # Too many labels
# # (More than 20 in non-time series data, more than 15 in UCR)
# TOO_MANY_LABELS = [
#     # artificial
#     "D31.arff", "fourty.arff",
#     # real-world
#     "cpu.arff", "letter.arff",
# ]

# # Too many samples
# # (More than 10000)
# TOO_MANY_SAMPLES = [
#     # artificial
#     "mopsi-finland.arff", "birch-rg3.arff", "birch-rg2.arff",
#     "birch-rg1.arff",
#     # real-world
#     "letter.arff",
# ]

def get_list_datasets_from_github(
        data_source: str = "artificial",
        with_unknown_k: bool = True,
        with_invalid: bool = True,
    ) -> List[str]:
    """
    Get the list of datasets from the GitHub repository.

    Parameters
    ----------
    data_source : str, optional
        Dataset list to fetch, typically ``"artificial"`` or
        ``"real-world"``.
    with_unknown_k : bool, optional
        If ``False``, exclude datasets whose number of clusters is not
        known in advance, except unimodal datasets.
    with_invalid : bool, optional
        If ``False``, exclude datasets listed in :data:`INVALID`.

    Returns
    -------
    List[str]
        Dataset filenames published in the remote repository after the
        requested filters are applied.
    """
    all_datasets = []
    for line in urllib.request.urlopen(f"{URL_ROOT}{data_source}.txt"):
        all_datasets.append(line.decode('utf-8').strip())

    if not with_unknown_k:
        all_datasets = [
            f for f in all_datasets
            if not (f in UNKNOWN_K and f not in UNIMODAL)
        ]
    if not with_invalid:
        all_datasets = [f for f in all_datasets if f not in INVALID]

    return all_datasets

def arff_from_github(url, verbose=False):
    """
    Load an ARFF dataset from a URL.

    Parameters
    ----------
    url : str
        URL pointing to an ARFF file.
    verbose : bool, optional
        If True, print the HTTP status code, by default False.

    Returns
    -------
    Tuple[Union[None, np.ndarray], Union[Exception, arff.MetaData]]
        Parsed ARFF data and metadata when successful. If loading fails,
        returns ``(None, exception)``.
    """
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if verbose:
                print(response.status, flush=True)
            arff_data = io.StringIO(response.read().decode('utf-8'))
            data, meta = arff.loadarff(arff_data)
    except Exception as ex:
        print(ex, flush=True)
        return None, ex
    return data, meta

def load_data_from_github(
    url: str,
    with_labels: bool = True
) -> Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]:
    """
    Return data, labels, and metadata from a GitHub ARFF URL.

    Non-numerical variables are ignored.

    Parameters
    ----------
    url : str
        URL of the dataset.
    with_labels : bool, optional
        If True, include labels from the ``class`` column, by default
        True.

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]
        Numeric data array, optional labels array, and ARFF metadata.
    """
    data, meta = arff_from_github(url)
    df = pd.DataFrame(data)
    # We keep only numerical variables
    data_col = [
        c for c, t in zip(df.columns, df.dtypes)
        if (str(c).lower() != "class") and t in ["float", "int"]
    ]
    class_col = [c for c in df.columns if str(c).lower() == "class"]
    # Get only data, not the labels and convert to numpy
    if with_labels:

        data = df[data_col].to_numpy()
        labels = df[class_col].to_numpy()
    else:
        data = df[data_col].to_numpy()
        labels = None
    return data, labels, meta



def get_data_labels(
    fname: str,
    url: str,
) -> Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]:
    """
    Get a dataset, labels, and metadata from GitHub.

    It is important to keep fname and url separate, as fname is used to
    check if the dataset is in the UNLABELED or UNIMODAL lists.

    The returned labels is ``None`` if the dataset had no labels originally
    provided, and if we can not a priori assume that the dataset is unimodal.

    Parameters
    ----------
    fname : str
        Dataset filename.
    url : str,
        URL for the dataset

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]
        Data array, optional labels, and ARFF metadata. Labels are
        synthesized as a single class for known unimodal datasets.
    """
    n_labels = None

    # If the dataset is in the UNLABELED list, we don't expect labels
    # But it could be unimodal, in which case we set n_labels to 1
    if fname in UNLABELED:
        with_labels = False       # No labels originally provided
        if fname in UNIMODAL:
            n_labels = 1
        else:
            n_labels = None
    else:
        with_labels = True        # Labels originally provided

    # Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]
    # labels is None if there were no labels originally provided
    data, labels, meta = load_data_from_github(
        url + fname, with_labels=with_labels
    )

    # If there were labels originally provided
    if with_labels:
        labels, n_labels = process_labels(labels)
    # If there were no labels provided but we know it's unimodal
    # Then create a unique class
    elif n_labels == 1:
        label_shape = (len(data), )
        labels = np.zeros(label_shape, dtype=int)

    return data, labels, meta

def save_data_labels_from_github(
    dataset_names: List[str],
    path_data: str = "./",
    data_source: str = "artificial",
) -> None:
    """
    Save data and labels from GitHub to CSV files.

    Parameters
    ----------
    dataset_names : List[str]
        Dataset filenames to download.
    path_data : str, optional
        Destination directory where ``*_data.csv`` and ``*_labels.csv``
        files will be written.
    data_source : str, optional
        Dataset collection to read from under :data:`URL_ROOT`.

    Returns
    -------
    None
        This function writes CSV files to disk and returns nothing.
    """
    os.makedirs(path_data, exist_ok=True)

    for d in dataset_names:
        data, labels, meta = get_data_labels(
            fname=d, url=f"{URL_ROOT}{data_source}/"
        )
        # labels = labels.astype(float)
        pd.DataFrame(labels).to_csv(
            f"{path_data}{d}_labels.csv".replace(".arff", ""),
            header=False, index=False,
        )
        pd.DataFrame(data).to_csv(
            f"{path_data}{d}_data.csv".replace(".arff", ""),
            header=False, index=False,
        )