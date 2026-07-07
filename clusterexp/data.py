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


def find_datasets(
    path_data:str,
    include_path_data: bool = False
) -> List[str]:
    """
    Find datasets in a given folder.

    - recursively go through the given folder
    - finds files that ends with `_data.csv` and `_labels.csv` or `_data.tsv` and `_labels.tsv` or `_data.npy` and `_labels.npy`
    - make sure that you do have both data and label files
    - returns a list of dataset names as ``[full/path/to/DATASET]``
      with the `_data.ext` but not the `_labels.ext` which is then easy to infer anyway. Originally I wanted to
      have a dictionnary with a shortname for the dataset (excluding the
      root of the path to the dataset but there could be issues if a
      given dataset has the same filename in several subfolders)

    This function assumes that the data is already formatted as
    expected with:
    - a file for the data and a file for the labels
    - they share the same extension (csv, tsv, or npy)


    Parameters
    ----------
    path_data : str
        Path to the folder containing datasets.
    include_path_data : bool, optional
        If ``True``, return dataset paths including the ``path_data``
        prefix. Otherwise return paths relative to ``path_data``.

    Returns
    -------
    List[str]
        Dataset paths whose data and label files both exist.
    """
    datasets = []
    extensions = [".csv", ".tsv", ".npy"]
    for root, dirs, files in os.walk(path_data):
        # Convert to Path objects for easier handling
        files = [Path(f) for f in files]

        # Find data and label files based on naming conventions
        data_files = [
            f for f in files
            if f.stem.endswith("_data") and f.suffix in extensions
        ]
        label_fnames = [
            f.name for f in files
            if f.stem.endswith("_labels") and f.suffix in extensions
        ]

        # Keep only data files that have a corresponding label file
        # We could directly replace _data without the extension but it's a bit
        # less safe, in case the pattern "_data" appears somewhere else
        kept_data_fnames = [
            f.name for f in data_files
            if f.name.replace(f"_data{f.suffix}", f"_labels{f.suffix}") in label_fnames
        ]

        # Add full paths to the resulting list
        for f in kept_data_fnames:
            datasets.append(os.path.join(root, f))

    # Remove the path_data prefix from all datasets
    if not include_path_data:
        datasets = [os.path.relpath(d, path_data) for d in datasets]

    return datasets

def load_data_labels(
    fname_data: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data and labels from CSV files.

    Parameters
    ----------
    fname_data : str
        Path to the data file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Data array and labels array loaded from matching ``*_data.ext`` and
        ``*_labels.ext`` files, where `ext` can be "csv", "tsv", or "npy".
    """
    # We could directly replace _data without the extension but it's a bit less
    # safe, in case the pattern "_data" appears somewhere else in the path
    ext = Path(fname_data).suffix
    fname_labels = fname_data.replace(f"_data{ext}", f"_labels{ext}")

    if ext in [".npy", "npy"]:
        data = np.load(fname_data)
        labels = np.load(fname_labels)
    elif ext in [".csv", "csv", ".tsv", "tsv"]:
        ext = ext.lstrip(".")
        data = pd.read_csv(fname_data, header=None).to_numpy()
        labels = pd.read_csv(fname_labels, header=None).to_numpy()
    else:
        raise ValueError(f"Unsupported extension: {ext}. Use 'csv', 'tsv' or 'npy'.")
    return data, labels

def filter_datasets(
        datasets:List[str],
        path_data:str = "",
        **constraints) -> Dict[str, List[str]]:
    """
    Filter datasets based on specified constraints.

    Parameters
    ----------
    datasets : List[str]
        List of dataset names to filter.
    path_data : str, optional
        Path to the folder containing datasets, by default "". By default, `datasets` are assumed to omit `path_data` prefix,
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
        Dictionary with ``kept_datasets`` and ``dropped_datasets``
        entries describing which datasets passed each constraint.
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
        data, labels = load_data_labels(f"{path_data}{dataset}")

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

def is_time_series(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Determine whether to use time series distance based on data shape.

    Potentially reshape the data if it has a single time step.

    Parameters
    ----------
    data : np.ndarray
        The input data array.

    Returns
    -------
    Tuple[np.ndarray, bool]
        The potentially reshaped data and a boolean indicating whether to use time series distance: ``True`` if data of shape ``(N, T, d)`` with ``T > 1``, ``False`` otherwise.
    """
    # Static data (N, d)
    if len(data.shape) == 2:
        (N, d) = data.shape
        ts_dist = False
    # Time series data (N, T, d)
    elif len(data.shape) == 3:
        (N, T, d) = data.shape
        if T == 1:
            ts_dist = False
            data = np.squeeze(data, axis=1)
        else:
            ts_dist = True
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    return data, ts_dist

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
    # Find unique classes and map them to integers
    labels = labels.flatten()
    classes = np.unique(labels)
    map_classes = {c:i for i,c in enumerate(classes)}
    n_labels = len(classes)

    if n_labels == N:
        n_labels = 1
        new_labels = np.zeros_like(labels, dtype=int)
    else:
        new_labels = np.array(
            [map_classes[label] for label in labels],
            dtype=int)
    return new_labels, n_labels