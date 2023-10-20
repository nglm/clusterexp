import io
import urllib.request
from scipy.io import arff
import pandas as pd
import numpy as np
import os
from typing import List, Dict

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
    "s-set3.arff", "s-set3.arff",
    # real-world
    "water-treatment.arff"
]

# Unknown number of clusters
UNKNOWN_K = [
    # artificial
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",
    # real-world
]

def arff_from_github(url, verbose=False):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if verbose:
                print(response.status, flush=True)
            arff_data = io.StringIO(response.read().decode('utf-8'))
            data, meta = arff.loadarff(arff_data)
    except Exception as ex:
        print(ex, flush=True)
    return data, meta

def load_data_from_github(url, with_labels=True):
    data, meta = arff_from_github(url)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.lower()
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

def get_data_labels(fname, path="./"):
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
        n_labels = len(np.unique(labels))
        if n_labels == N:
            n_labels = 1
            labels = np.zeros(N)
    return data, labels, n_labels, meta

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