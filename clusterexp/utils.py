import io
import urllib.request
from scipy.io import arff
import pandas as pd
import numpy as np
import os
import json

from typing import List, Dict, Tuple, Union

# --------------------- ClusteringBenchmark ----------------------------

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'

N_SAMPLES_MAX = 10000

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

# Too many labels
# (More than 20 in non-time series data, more than 15 in UCR)
TOO_MANY_LABELS = [
    # real-world
    "arrhythmia.arff", "water-treatment.arff", "wdbc.arff",
    "dermatology.arff", "sonar.arff", "german.arff", "iono.arff",
]

# Too many labels
# (More than 20 in non-time series data, more than 15 in UCR)
TOO_MANY_SAMPLES = [
    # artificial
    "mopsi-finland.arff", "birch-rg3.arff", "birch-rg2.arff",
    "birch-rg1.arff",

    # real-world
    "letter.arff",
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


# ---------------------------- UCR -------------------------------------
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

def arff_from_github(url, verbose=False):
    """
    Returns data as arff and metadata if no exceptions were found.

    If an exception was found (e.g. "String attributes not supported
    yet, sorry"), then returns None for the data and the error message
    as the meta data.
    """
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if verbose:
                print(response.status, flush=True)
            arff_data = io.StringIO(response.read().decode('utf-8'))
            data, meta = arff.loadarff(arff_data)
    except Exception as ex:
        if verbose:
            print(ex, flush=True)
        return None, ex
    return data, meta

def print_heads(
    fnames: List[str],
    path:str = "./",
    n_labels_max:int = 20,
    n_samples_max:int = 10000,
) -> None:
    """
    Print all dataframe headers from a list of filenames and path.

    :param fnames: list of filenames
    :type fnames: List[str]
    :param path: path to the directory containing all files, defaults to
        "./"
    :type path: str, optional
    """
    summary = {}
    for f in fnames:
        summary[f] = {}
        url = path + f
        print(url)
        data, meta = arff_from_github(url)
        # Print the head of the data frame, to get a better idea of the
        # dataset
        if data is not None:
            df = pd.DataFrame(data)
            cols = df.columns.str.lower()

            labeled = "class" in cols
            has_na = df.isnull().sum().sum() > 0
            shape = (len(df), len(cols))

            msg = (
                f"{shape}\n" +
                f"Labeled:         {labeled}\n" +
                f"Has NA values:   {has_na}\n" +
                f"Too many labels: {shape[1]>n_labels_max}\n" +
                f"Too many samples:{shape[0]>n_samples_max}"
            )
            print(msg)
            print(df.head())

            summary[f]["labeled"] = labeled
            summary[f]["has_na"] = has_na
            summary[f]["shape"] = shape
        # If there was a problem loading the data, then the error message
        # is returned in "meta"
        else:
            summary[f]["error"] = meta
            print(meta)
    return summary


def load_data_from_github(
    url: str,
    with_labels: bool = True
) -> Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]:
    """
    Return data, labels and metadata from github url

    Ignore non-numerical variables in the datasets

    :param url: github url of the dataset
    :type url: str
    :param with_labels: _description_, defaults to True
    :type with_labels: bool, optional
    :return: _description_
    :rtype: Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]
    """
    data, meta = arff_from_github(url)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.lower()
    # We keep only numerical variables
    data_col = [
        c for c, t in zip(df.columns, df.dtypes)
        if (c != "class") and t in ["float", "int"]
    ]
    # Get only data, not the labels and convert to numpy
    if with_labels:

        data = df[data_col].to_numpy()
        labels = df["class"].to_numpy()
    else:
        data = df[data_col].to_numpy()
        labels = None
    return data, labels, meta

def process_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Give the real number of labels and labels in case of n_labels = N
    """
    N = len(labels)
    n_labels = len(np.unique(labels))
    if n_labels == N:
        n_labels = 1
        labels = np.zeros(N)
    return labels, n_labels

def get_data_labels(
    fname: str,
    path: str ="./"
) -> Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]:
    """
    Get dataset, labels, number of labels, and metadata for non UCR data

    :param fname: Filename of the dataset
    :type fname: str
    :param path: Path to the file, defaults to "./"
    :type path: str, optional
    :return: All information about the dataset, the data, labels, number
        of labels, and metadata
    :rtype: Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]
    """
    n_labels = None
    if fname in UNLABELED:
        with_labels = False
        if fname in UNIMODAL:
            n_labels = 1
        else:
            n_labels = None
    else:
        with_labels = True
    data, labels, meta = load_data_from_github(
        path + fname, with_labels=with_labels
    )
    N = len(data)
    if with_labels:
        labels, n_labels = process_labels(labels)
    return data, labels, n_labels, meta

def get_data_labels_UCR(
    fname: str,
    path: str ="./"
) -> Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]:
    """
    Get dataset, labels number of labels, and metadata for UCR data

    :param fname: Filename of the dataset
    :type fname: str
    :param path: Path to the file, defaults to "./"
    :type path: str, optional
    :return: All information about the dataset, the data, labels, number
        of labels, and metadata
    :rtype: Tuple[np.ndarray, Union[None, np.ndarray], int, None]
    """

    df = pd.read_csv(fname, sep="\t")

    data = np.expand_dims(df.iloc[:, 1:].to_numpy(), axis=2)

    labels = df.iloc[:, 0].to_numpy()
    labels, n_labels = process_labels(labels)

    return data, labels, n_labels, None

def get_list_datasets(fname: str) -> List[str]:
    """
    Read the file containing the list of dataset names

    :param fname: name of the file with the list of dataset names
    :type fname: str
    :return: the list of dataset names
    :rtype: List[str]
    """
    with open(fname) as f:
        datasets = f.read().splitlines()
    return datasets

def get_list_exp(
    dataset_name: str,
    res_dir: str = './res/',
) -> List[str]:
    """
    For each dataset, find all experiments working on this dataset
    but using different clustering methods (by filtering on the
    filename)

    :param dataset_name: Name of the dataset
    :type dataset_name: str
    :return: List of experiment filenames
    :rtype: List[Dict]
    """
    # List of directories, corresponding to clustering methods
    list_dirs = [
        dname.strip() for dname in next(os.walk(res_dir))[1]
        if dname not in ["Selected"]]

    fnames = []
    for dir in list_dirs:

        dir_fnames = [f.name for f in os.scandir(res_dir + dir)]
        # get full directory + filename without the extension
        fnames += [
            dir + "/" + fname[:-5] for fname in dir_fnames
            if dataset_name + ".json" in fname
        ]
    return fnames

def load_json(fname: str) -> Dict:
    def object_hook(json_dict):
        return {
            int(k) if k.isdigit() else k: v
            for (k, v) in json_dict.items()
        }
    with open(fname) as f_json:
        d = json.load(f_json, object_hook=object_hook)
    return d

def write_list_datasets(fname:str, lines: List[str]) -> None:
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))

def get_fname(
    d: str,
    only_root: bool=False,
    data_source: str = "artificial",
) -> str:
    """
    Find the filename (or root) corresponding to the UCR dataset

    :param d: dataset name
    :type d: str
    :param only_root: defaults to False
    :type only_root: bool, optional
    :param data_source: Dataset source ("UCR", "artificial" or
        "real-world")
    :type data_source: str
    :return: The filename (or root) corresponding to the dataset
    :rtype: str
    """
    fname = ""
    if data_source == "UCR":
        if d in ILL_FORMATED:
            fname += f"{ILL_FORMATED_DIR}"
        fname += f"{d}/"
        if not only_root:
            fname += f"{d}_TRAIN.tsv"
    else:
        fname += f"{d}"
    return fname
