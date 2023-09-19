import io
import urllib.request
from scipy.io import arff
import pandas as pd
import numpy as np

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

# Just one cluster
UNIMODAL = [
    "birch-rg1.arff", "birch-rg2.arff",
    "golfball.arff",
]
# No class column in the data
UNLABELED = [
    "birch-rg1.arff", "birch-rg2.arff",
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set3.arff",
]

# Unknown number of clusters
UNKNOWN_K = [
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",
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
    # Get only data, not the labels and convert to numpy
    if with_labels:
        data = df.iloc[:, 0:-1].to_numpy()
        labels = df.iloc[:, -1].to_numpy()
    else:
        data = df.to_numpy()
        labels = None
    return data, labels, meta

def get_data_labels(fname, path=PATH):
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