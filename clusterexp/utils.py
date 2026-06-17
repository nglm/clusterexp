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
    # artificial
    "D31.arff", "fourty.arff",

    # real-world
    "cpu.arff", "letter.arff",

    # UCR
    "PigArtPressure", "FiftyWords", "Adiac", "PigCVP", "Phoneme",
    "PigAirwayPressure", "WordSynonyms", "NonInvasiveFetalECGThorax1",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "Crop", "NonInvasiveFetalECGThorax2", "ShapesAll",
]

# Too many samples
# (More than 10000)
TOO_MANY_SAMPLES = [
    # artificial
    "mopsi-finland.arff", "birch-rg3.arff", "birch-rg2.arff",
    "birch-rg1.arff",

    # real-world
    "letter.arff",

    # UCR
    "ElectricDevices", "Crop", "FordA", "FordB"
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

def print_heads(
    fnames: List[str],
    path:str = "./",
    n_labels_max:int = 20,
    n_samples_max:int = 10000,
    UCR: bool = False,
) -> None:
    """
    Print summary information and heads for multiple datasets.

    Parameters
    ----------
    fnames : List[str]
        Dataset names or filenames.
    path : str, optional
        Prefix used to resolve each dataset location, by default "./".
    n_labels_max : int, optional
        Threshold used to flag datasets with too many labels,
        by default 20.
    n_samples_max : int, optional
        Threshold used to flag datasets with too many samples,
        by default 10000.
    UCR : bool, optional
        If True, load UCR-formatted files from local TSV paths;
        otherwise load ARFF datasets, by default False.

    Returns
    -------
    Dict[str, Dict]
        Per-dataset summary containing metadata such as shape,
        labeling information, and potential loading errors.
    """
    print(f"MAX LABELS: {n_labels_max}\nMAX SAMPLES: {n_samples_max}\n")
    summary = {}
    for f in fnames:
        summary[f] = {}

        # Get the dataframe corresponding to the filename
        # We don't use get_data_labels functions here because we want to
        # use the raw df.
        if UCR:
            fname = get_fname(f, only_root=False, data_source='UCR')
            print(fname)
            full_f = path+fname
            try:
                df = pd.read_csv(full_f, sep="\t")
            except Exception as ex:
                meta = ex
                df = None
        else:
            full_f = path + f
            print(full_f)
            data, meta = arff_from_github(full_f)
            if data is None:
                df = None
            else:
                df = pd.DataFrame(data)
        # Print the head of the data frame, to get a better idea of the
        # dataset

        if df is not None:
            cols = df.columns.str.lower()

            labeled = (("class" in cols) or UCR)
            has_na = df.isnull().sum().sum() > 0
            shape = (len(df), len(cols))


            # We use get_data_labels here just to count the labels,
            # not to get df as it would already be processed
            if UCR:
                _, _, n_labels, _ = get_data_labels_UCR(full_f, path="")
            else:
                if "class" in cols:
                    _, _, n_labels, _ = get_data_labels(full_f, path="")
                else:
                    n_labels = None

            if labeled:
                too_many_labels = n_labels > n_labels_max
            else:
                too_many_labels = False

            msg = (
                f"Shape: {shape}   |   n_labels: {n_labels}\n" +
                f"Labeled:         {labeled}\n" +
                f"Has NA values:   {has_na}\n" +
                f"Too many labels: {too_many_labels}\n" +
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
    Return data, labels, and metadata from an GitHub ARFF URL.

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
    Encode labels and infer the effective count.

    Parameters
    ----------
    labels : np.ndarray
        Original label values.

    Returns
    -------
    Tuple[np.ndarray, int]
        Encoded labels and the number of effective classes. If each
        sample has a unique label, labels are collapsed to one class.
    """
    N = len(labels)
    classes = np.unique(labels)
    map_classes = {c:i for i,c in enumerate(classes)}
    n_labels = len(classes)
    if n_labels == N:
        n_labels = 1
        labels = np.zeros_like(labels, dtype=int)
    else:
        labels = np.array(
            [map_classes[label] for label in labels],
            dtype=int)
    return labels, n_labels

def get_data_labels(
    fname: str,
    path: str ="./"
) -> Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]:
    """
    Get dataset, labels, number of labels, and metadata for non UCR data

    Parameters
    ----------
    fname : str
        Dataset filename.
    path : str, optional
        Prefix path or URL for the dataset, by default "./".

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]
        Data array, optional labels, inferred number of labels, and
        ARFF metadata.
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
    if with_labels:
        labels, n_labels = process_labels(labels)
    return data, labels, n_labels, meta

def get_data_labels_UCR(
    fname: str,
    path: str ="./"
) -> Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]:
    """
    Get dataset, labels number of labels, and metadata for UCR data

    Parameters
    ----------
    fname : str
        Path to the UCR TSV file.
    path : str, optional
        Unused. Kept for API consistency with other loaders,
        by default "./".

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], int, None]
        Time-series data reshaped as ``(N, T, 1)``, encoded labels,
        number of labels, and ``None`` metadata.
    """

    df = pd.read_csv(fname, sep="\t")

    data = np.expand_dims(df.iloc[:, 1:].to_numpy(), axis=2)

    labels = df.iloc[:, 0].to_numpy()
    labels, n_labels = process_labels(labels)

    return data, labels, n_labels, None

def get_list_datasets(fname: str) -> List[str]:
    """
    Read a file containing one dataset name per line.

    Parameters
    ----------
    fname : str
        Path to the file with dataset names.

    Returns
    -------
    List[str]
        Dataset names.
    """
    with open(fname) as f:
        datasets = f.read().splitlines()
    return datasets

def get_list_exp(
    dataset_name: str,
    res_dir: str = './res/',
    suffix: str = ".json",
) -> List[str]:
    """
    For each dataset, find all experiments working on this dataset

    Each experiment on a dataset used a different clustering method.
    This function filters based on the filename of the experiment file.

    The extension (".json") is not included in the returned filenames

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    res_dir : str, optional
        Path to the directory containing the experiments, by default
        './res/'
    suffix : str, optional
        Suffix of the experiment filenames, by default ".json", but
        using "_scored.json" can be useful to use score files instead of
        clustering files.

    Returns
    -------
    List[str]
        List of experiment filenames (excluding the extension ".json")
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
            if dataset_name + suffix in fname
        ]
    return fnames

def load_json(fname: str) -> Dict:
    """
    Load a JSON file and cast numeric string keys to integers.

    Parameters
    ----------
    fname : str
        Path to the JSON file.

    Returns
    -------
    Dict
        Parsed JSON dictionary with digit-only keys converted to
        integers.
    """
    def object_hook(json_dict):
        return {
            int(k) if k.isdigit() else k: v
            for (k, v) in json_dict.items()
        }
    with open(fname) as f_json:
        d = json.load(f_json, object_hook=object_hook)
    return d

def write_list_datasets(fname:str, lines: List[str]) -> None:
    """
    Write dataset names to a text file, one per line.

    Parameters
    ----------
    fname : str
        Output file path.
    lines : List[str]
        Dataset names to write.
    """
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))

def get_fname(
    d: str,
    only_root: bool=False,
    data_source: str = "artificial",
) -> str:
    """
    Find the filename (or root) corresponding to the UCR dataset

    Parameters
    ----------
    d : str
        Dataset name.
    only_root : bool, optional
        If True and ``data_source == 'UCR'``, return only the dataset
        directory path, by default False.
    data_source : str, optional
        Dataset source: ``"UCR"``, ``"artificial"``, or
        ``"real-world"``, by default ``"artificial"``.

    Returns
    -------
    str
        Dataset filename or root path depending on ``data_source`` and
        ``only_root``.
    """
    fname = ""
    if data_source == "UCR":
        if d in ILL_FORMATED:
            fname += f"{ILL_FORMATED_DIR}"
        fname += f"{d}/"
        if not only_root:
            fname += f"{d}_TRAIN.tsv"
    else:
        fname = f"{d}"
    return fname
