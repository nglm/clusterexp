"""Additional util fonctions not related to datasets, config or plots"""

import io
import urllib.request
from scipy.io import arff
import pandas as pd
import numpy as np
import os
import json

from typing import List, Dict, Tuple, Union

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

def write_json(fname: str, data: Dict) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    fname : str
        Path to the JSON file.
    data : Dict
        Dictionary to write to the JSON file.
    """
    json_str = json.dumps(data, indent=2)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(json_str)

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
