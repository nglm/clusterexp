"""Additional util fonctions not related to datasets, config or plots"""


import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import inspect
import importlib

from typing import List, Dict, Tuple, Union, Sequence, Any

# def get_list_exp(
#     dataset_name: str,
#     res_dir: str = './res/',
#     suffix: str = ".json",
# ) -> List[str]:
#     """
#     For each dataset, find all experiments working on this dataset

#     Each experiment on a dataset used a different clustering method.
#     This function filters based on the filename of the experiment file.

#     The extension (".json") is not included in the returned filenames

#     Parameters
#     ----------
#     dataset_name : str
#         Name of the dataset
#     res_dir : str, optional
#         Path to the directory containing the experiments, by default
#         './res/'
#     suffix : str, optional
#         Suffix of the experiment filenames, by default ".json", but
#         using "_scored.json" can be useful to use score files instead of
#         clustering files.

#     Returns
#     -------
#     List[str]
#         List of experiment filenames (excluding the extension ".json")
#     """

#     # List of directories, corresponding to clustering methods
#     list_dirs = [
#         dname.strip() for dname in next(os.walk(res_dir))[1]
#         if dname not in ["Selected"]]

#     fnames = []
#     for dir in list_dirs:

#         dir_fnames = [f.name for f in os.scandir(res_dir + dir)]
#         # get full directory + filename without the extension
#         fnames += [
#             dir + "/" + fname[:-5] for fname in dir_fnames
#             if dataset_name + suffix in fname
#         ]
#     return fnames


def get_obj_from_string(obj_str: str) -> Any:
    """
    Return the object corresponding to a string representing it.

    The object can be a class or a function. The string must be of the
    form `"package.module.class"` or `"package.module.function"`.

    Both the hidden and non-hidden module names are supported.
    The hidden module name is the one that is used when importing a
    class or function

    The corresponding class or function will be imported.

    If the string is not of the correct form, an ImportError will be
    raised.

    Parameters
    ----------
    obj_str : str
        String representing the object to be imported.

    Returns
    -------
    Any
        The object corresponding to the string.
    """
    # rsplit(".", 1) will split once, at the very last occurence
    module_path, obj_name = obj_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def class_to_string(cls):
    """
    Return the string corresponding to a class.

    Note that this can yield the "hidden" class of an object, with
    hidden module names
    """
    return f"{cls.__module__}.{cls.__name__}"

def obj_to_string(obj):
    cls = obj.__class__
    return f"Instance of {cls.__module__}.{cls.__name__}"

def serialize(obj):
    """
    Serialize an object to a JSON-compatible format.
    """
    # To check if the object is serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, Sequence):
            res = [serialize(x) for x in obj]
        # If we are dealing with a dict of object
        elif isinstance(obj, dict):
            res = {key : serialize(item) for key, item in obj.items()}
        # If we are dealing with a dict of object
        elif isinstance(obj, np.ndarray):
            res = obj.tolist()
        else:
            # True for classes/types, False for instances
            if inspect.isclass(obj):
                res = class_to_string(obj)
            elif inspect.isfunction(obj):
                res = f"{obj.__module__}.{obj.__name__}"
            else:
                res = obj_to_string(obj)
        return res


def simplify_dict(config:dict) -> dict:
    """
    Translate a given dict to a jsonable dict

    Classes and functions will be written as package.module.class
    """
    #normal_types = [Sequence, str, list, dict, int, float, bool, type(None)]

    simpler_dict = {}
    for k, v in config.items():
        # Serialize value if necessary
        simpler_dict[k] = serialize(v)
    return simpler_dict

def interpret_saved_dict(config:dict) -> dict:
    """
    Translate a given dict to a config dict, using classes and functions

    Make sure that classes and functions are written as package.module.class

    This function will also add the default values to the config.
    """

    interpreted_dict = {}
    for k, v in config.items():
        # If the value is a string, try to interpret it as a class or function
        if isinstance(v, str):
            try:
                interpreted_dict[k] = get_obj_from_string(v)
            # If there was an error, it's probably because this was a regular
            # string, not a string representing a class or function
            except (ImportError, AttributeError, ValueError):
                interpreted_dict[k] = v
        # If the value is a dict, recursively interpret it
        elif isinstance(v, dict):
            interpreted_dict[k] = interpret_saved_dict(v)
        # Else, assume that the format is correct and keep the value as is
        else:
            interpreted_dict[k] = v

    return interpreted_dict

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
    p = Path(fname)
    p.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(data, indent=2)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(json_str)

# def get_fname(
#     d: str,
#     only_root: bool=False,
#     data_source: str = "artificial",
# ) -> str:
#     """
#     Find the filename (or root) corresponding to the UCR dataset

#     Parameters
#     ----------
#     d : str
#         Dataset name.
#     only_root : bool, optional
#         If True and ``data_source == 'UCR'``, return only the dataset
#         directory path, by default False.
#     data_source : str, optional
#         Dataset source: ``"UCR"``, ``"artificial"``, or
#         ``"real-world"``, by default ``"artificial"``.

#     Returns
#     -------
#     str
#         Dataset filename or root path depending on ``data_source`` and
#         ``only_root``.
#     """
#     fname = ""
#     if data_source == "UCR":
#         if d in ILL_FORMATED:
#             fname += f"{ILL_FORMATED_DIR}"
#         fname += f"{d}/"
#         if not only_root:
#             fname += f"{d}_TRAIN.tsv"
#     else:
#         fname = f"{d}"
#     return fname

def save_log(
        log_fname: str,
        log_dict: dict[str, Any],
        overwrite: bool = False,
        add_date: bool = True,
        new_name: bool = False,
        verbose: bool = False,
    ) -> str | None:
    """Save a JSON log file and optionally rename it when a file exists.

    Parameters
    ----------
    log_fname : str
        Target file path for the log file, typically a ``.json`` filename.
    log_dict : dict[str, Any]
        Dictionary of log metadata to serialize. Typical keys include run
        parameters, status flags, and summary values. The ``log_filename`` key
        is added or updated before writing.
    overwrite : bool, default=False
        Whether to overwrite an existing log file at ``log_fname``.
    add_date : bool, default=True
        Whether to append a ``YYYY-MM-DD--HH:MM:SS`` timestamp to the filename
        before saving.
    new_name : bool, default=False
        Whether to create a new timestamped filename when ``log_fname`` already
        exists and ``overwrite`` is ``False``.
    verbose : bool, default=False
        Whether to print status messages during save operations.

    Returns
    -------
    str | None
        The final log filename that was written, or ``None`` if saving was
        skipped.
    """

    if add_date:
        # Create a new filename by adding a suffix to the original filename
        ext = Path(log_fname).suffix
        base = log_fname.split(ext)[0]
        full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
        log_fname = f"{base}-{full_date}{ext}"

    if os.path.isfile(log_fname):

        # Common message if log exists
        if int(verbose) > 0:
            print(f"Log file {log_fname} already exists.")
        # Added message if log should be overwritten
        if overwrite:
            if int(verbose) > 0:
                print(f"Overwriting log file {log_fname}.")
        # Added message if we then cancel the saving of the log file
        elif not new_name:
            log_fname = None
            if int(verbose) > 0:
                print(f"Not overwriting nor creating new filename. Skipping log saving.")

        # If we create a new name
        else:

            # Create a new filename by adding a suffix to the original filename
            ext = Path(log_fname).suffix
            base = log_fname.split(ext)[0]
            full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
            log_fname = f"{base}-{full_date}{ext}"

            if int(verbose) > 0:
                print(f"Creating new filename: {log_fname}.")

    if log_fname is not None:

        # Make sure path exists otherwise create it
        p = Path(log_fname)
        p.parent.mkdir(parents=True, exist_ok=True)

        log_dict["log_filename"] = log_fname

        log_dict_serializable = simplify_dict(log_dict)

        with open(log_fname, 'w') as f_log:
            json.dump(log_dict_serializable, f_log, indent=2)

        if int(verbose) > 0:
            print(f"Saved log file to {log_fname}.")

    return log_fname


def print_log(
        log:dict,
    ) -> None:
    """
    Print a log dict in a readable format
    """
    simpler_dict = simplify_dict(log)
    print(f"\n┌─{'─'*70}─┐")
    print(f"{" "*3} Log file: {log['log_data']['log_fname']}")
    print(f"START LOG")
    print(json.dumps(simpler_dict, indent=2), flush=True)
    print(f"END LOG")
    print(f"\n└─{'─'*70}─┘", flush=True)

def extract_log_from_text(fname) -> list[dict]:
    """
    Extract log dicts from a log text file.

    The text file should contain a JSON string representing the log dict,
    starting with a line containing "START LOG" and ending with a line
    containing "END LOG".

    There could be several logs in the same text file, each one starting with "START LOG" and ending with "END LOG", but logically, there should be only 2 logs per text file, the one at the very beginning and the one at the very end, with some other text in between.

    Parameters
    ----------
    fname : str
        Path to the log text file.

    Returns
    -------
    l_logs : list[dict]
        A list of log dicts extracted from the text file.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    l_i_start = [i for i, line in enumerate(lines) if line == "START LOG\n"]
    l_i_end = [i for i, line in enumerate(lines) if line == "END LOG\n"]
    l_logs = []
    for i_start, i_end in zip(l_i_start, l_i_end):
        json_lines = [l for l in lines[i_start + 1:i_end]]
        json_str = "".join(json_lines)
        json_dict = json.loads(json_str)
        l_logs.append(interpret_saved_dict(json_dict))
    return l_logs


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