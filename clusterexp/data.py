"""
General functions on datasets

This module assumes that the data is already formatted as expected, with a file for the data and a file for the labels. For functions specific to Barton's datasets or the UCR dataset, see barton.py or ucr.py.

Functions defined here are general to all datasets.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

from typing import List, Dict, Tuple, Union

# def print_heads(
#     fnames: List[str],
#     path:str = "./",
#     n_labels_max:int = 20,
#     n_samples_max:int = 10000,
#     UCR: bool = False,
# ) -> None:
#     """
#     Print summary information and heads for multiple datasets.

#     Parameters
#     ----------
#     fnames : List[str]
#         Dataset names or filenames.
#     path : str, optional
#         Prefix used to resolve each dataset location, by default "./".
#     n_labels_max : int, optional
#         Threshold used to flag datasets with too many labels,
#         by default 20.
#     n_samples_max : int, optional
#         Threshold used to flag datasets with too many samples,
#         by default 10000.
#     UCR : bool, optional
#         If True, load UCR-formatted files from local TSV paths;
#         otherwise load ARFF datasets, by default False.

#     Returns
#     -------
#     Dict[str, Dict]
#         Per-dataset summary containing metadata such as shape,
#         labeling information, and potential loading errors.
#     """
#     print(f"MAX LABELS: {n_labels_max}\nMAX SAMPLES: {n_samples_max}\n")
#     summary = {}
#     for f in fnames:
#         summary[f] = {}

#         # Get the dataframe corresponding to the filename
#         # We don't use get_data_labels functions here because we want to
#         # use the raw df.
#         if UCR:
#             fname = get_fname(f, only_root=False, data_source='UCR')
#             print(fname)
#             full_f = path+fname
#             try:
#                 df = pd.read_csv(full_f, sep="\t")
#             except Exception as ex:
#                 meta = ex
#                 df = None
#         else:
#             full_f = path + f
#             print(full_f)
#             data, meta = arff_from_github(full_f)
#             if data is None:
#                 df = None
#             else:
#                 df = pd.DataFrame(data)
#         # Print the head of the data frame, to get a better idea of the
#         # dataset

#         if df is not None:
#             cols = df.columns.str.lower()

#             labeled = (("class" in cols) or UCR)
#             has_na = df.isnull().sum().sum() > 0
#             shape = (len(df), len(cols))


#             # We use get_data_labels here just to count the labels,
#             # not to get df as it would already be processed
#             if UCR:
#                 _, _, n_labels, _ = get_data_labels_UCR(full_f, path="")
#             else:
#                 if "class" in cols:
#                     _, _, n_labels, _ = get_data_labels(full_f, path="")
#                 else:
#                     n_labels = None

#             if labeled:
#                 too_many_labels = n_labels > n_labels_max
#             else:
#                 too_many_labels = False

#             msg = (
#                 f"Shape: {shape}   |   n_labels: {n_labels}\n" +
#                 f"Labeled:         {labeled}\n" +
#                 f"Has NA values:   {has_na}\n" +
#                 f"Too many labels: {too_many_labels}\n" +
#                 f"Too many samples:{shape[0]>n_samples_max}"
#             )
#             print(msg)
#             print(df.head())

#             summary[f]["labeled"] = labeled
#             summary[f]["has_na"] = has_na
#             summary[f]["shape"] = shape
#         # If there was a problem loading the data, then the error message
#         # is returned in "meta"
#         else:
#             summary[f]["error"] = meta
#             print(meta)
#     return summary


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
    p = Path(fname)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(fname, 'w') as f:
        f.write('\n'.join(lines))


def find_datasets(path_data:str) -> List[str]:
    """
    Find datasets in a given folder.

    - recursively go through the given folder
    - finds files that ends with `_data.csv` and `_labels.csv`
    - extract the dataset name from the data file
    - make sure that you do have both files
    - returns a list of dataset names as ``[full/path/to/DATASET]``
      without the `_data.csv` and `_labels.csv`. Originally I wanted to
      have a dictionnary with a shortname for the dataset (excluding the
      root of the path to the dataset but there could be issues if a
      given dataset has the same filename in several subfolders)

    This function assumes that the data is already formatted as
    expected with:
    - a file for the data and a file for the labels,


    Parameters
    ----------
    path_data : str
        Path to the folder containing datasets.

    Returns
    -------
    List[str]
        List of dataset names found in the folder.
    """
    datasets = []
    for root, dirs, files in os.walk(path_data):
        data_files = [f for f in files if f.endswith("_data.csv")]
        label_files = [f for f in files if f.endswith("_labels.csv")]

        # Extract dataset names from data files
        data_names = {f.split('_data.csv')[0] for f in data_files}
        label_names = {f.split('_labels.csv')[0] for f in label_files}

        # Find common dataset names that have both data and labels
        common_names = data_names.intersection(label_names)

        # Add full paths to the datasets list
        for name in common_names:
            datasets.append(os.path.join(root, name))

    return datasets

def load_data_labels(
    fname_root: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data and labels from CSV files.

    Parameters
    ----------
    fname_root : str
        Root filename (without `_data.csv` or `_labels.csv` suffix).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Data array and labels array.
    """
    data = pd.read_csv(f"{fname_root}_data.csv", header=None).to_numpy()
    labels = pd.read_csv(f"{fname_root}_labels.csv", header=None).to_numpy()
    return data, labels

def filter_datasets(datasets:List[str], **constraints) -> Dict[str, List[str]]:
    """
    Filter datasets based on specified constraints.

    Parameters
    ----------
    datasets : List[str]
        List of dataset names to filter.
    **constraints : dict
        Constraints to apply for filtering. Possible keys include:
        - 'max_n_samples': Maximum number of samples allowed.
        - 'max_n_labels': Maximum number of labels allowed.
        - 'max_n_dims': Maximum number of dimensions allowed.
        - 'exclude': List of dataset names (or patterns) to exclude from the results.
        - 'include_only': List of dataset names (or patterns) to include in the results.

    Returns
    -------
    Dict[str, List[str]]
        Filtered dictionary of dataset names that meet the specified constraints.
    """
    dropped ={
            "max_n_samples": [],
            "max_n_labels": [],
            "max_n_dims": [],
            "exclude": [],
            "include_only": [],
    }
    kept = []

    for dataset in datasets:
        # Load the dataset to get its properties
        data, labels = load_data_labels(dataset)

        data_shape = data.shape
        n_samples = data_shape[0]
        n_dims = data_shape[-1]
        n_labels = len(np.unique(labels))

        keep = True

        # -------------  Apply constraints ---------------
        if n_samples > constraints.get('max_n_samples', float('inf')):
            dropped["max_n_samples"].append(dataset)
            keep = False
        if n_labels > constraints.get('max_n_labels', float('inf')):
            dropped["max_n_labels"].append(dataset)
            keep = False
        if n_dims > constraints.get('max_n_dims', float('inf')):
            dropped["max_n_dims"].append(dataset)
            keep = False
        if any(excl in dataset for excl in constraints.get('exclude', [])):
            dropped["exclude"].append(dataset)
            keep = False
        include_only = constraints.get('include_only', [])
        if include_only and not any(incl in dataset for incl in include_only):
            dropped["include_only"].append(dataset)
            keep = False

        if keep:
            kept.append(dataset)

    filtered_datasets = {
        "kept_datasets": kept,
        "dropped_datasets": dropped,
    }

    return filtered_datasets