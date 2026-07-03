import os
import sys
from datetime import datetime
import numpy as np
import time
from pycvi.cluster import get_clustering, generate_all_clusterings

from .config import load_config_as_dict, get_models_config
from .data import find_datasets, filter_datasets, load_data_labels, is_time_series
from .utils import save_log, print_log, interpret_saved_dict
from .clustering import compute_VI_quality, filter_experiments

from typing import Union

def prepare_data(config_fname:str) -> dict:
    """
    - Reads the config file: function `load_config_as_dict`
    - Prints the initiated log file containing the config file used `print_log` function
    - Find all datasets in the path_data folder (using the function `find_datasets`)
    - filter datasets based on the path_data and constraints defined in the config file (using the function `filter_datasets`)
      - Creates a list of kept datasets
      - Creates the dictionnary of sorted (kept, and one key per constraint) datasets
    - Creates a json resulting logfile `log-data-20XX-XX-XX.json`, with
      - One key `config_data` with the corresponding dict of the data config file used
      - One key `log_data` with the corresponding dict:
          - `dropped_datasets` a dict of lists of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `kept_datasets` list of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `log_fname` : `path/to/res/log-data-20XX-XX-XX` (without the `.txt` or `.json`)
    - Save the merged dictionary ``log-data-20XX-XX-XX.json``  with the function `save_log`
    - Save output log file ``log-data-20XX-XX-XX.txt``
    - Returns the merged dictionary ``log-data-20XX-XX-XX.json`` as a dict
    """
    # ---------------- Read config file ---------------------
    config = load_config_as_dict(config_fname)
    if "config_data" not in config:
        raise ValueError("The config file must contain a 'config_data' key.")
    path_data = config['config_data']['path_data']
    path_res = config['config_data']['path_res']

    # ----------- Prepare current log files ------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    log_fname = f'{path_res}log-data-{full_date}'
    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        "config_data": config['config_data'],
        "log_data": {
            "log_fname": log_fname,
        }}

    # Make the log visible in the output file
    print_log(log)

    # ---------------- Find datasets ------------------------
    datasets = find_datasets(path_data)

    # --------------- Filter datasets ----------------------
    constraints = config['config_data'].copy()
    constraints.pop('path_data', None)
    constraints.pop('path_res', None)

    filtered_datasets = filter_datasets(datasets, **constraints)

    # -------------------- Finalize log and save --------------
    log['log_data'].update(filtered_datasets)

    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )

    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log



def create_clusterings(config_fname:str, log_data_fname:Union[str, None] = None) -> dict:
    """
    Create clustering experiments for each dataset in the log file

    - Extract clustering model configuration from the config file
    - Get list of kept datasets based on log_data
    - for each dataset
        - Load data and labels
        - Get the true clusters from the labels
        - Decide whether whether to use ts_dist based on data shape
        for each clustering model in the config file:
            - instanciate scaler
            - call pycvi generate all clusterings
                - data
                - a model class
                - n_clusters_range
                - ts_dist (based on data shape)
                - model_kw
                - fit_predict_kw
                - an instanciated scaler (should be instanciated beforehand)
                - verbose
            - save clustering file
                - clusterings
                - VI
                - quality
    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = load_config_as_dict(config_fname)
    if "config_clustering" not in config:
        raise ValueError("The config file must contain a 'config_clustering' key.")

    if log_data_fname is None:
        log_data = prepare_data(config_fname)
    else:
        log_data = interpret_saved_dict(log_data_fname)

    path_data = log_data['config_data']['path_data']
    path_res = log_data['config_data']['path_res']

    # ----------- Prepare current log files ---------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    log_fname = f'{path_res}log-clustering-{full_date}'
    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        **log_data,
        "config_clustering": config['config_clustering'],
        "log_clustering": {
            "log_fname": log_fname,
        }}

    # Make the log visible in the output file
    print_log(log)

    # ================ Create clustering experiments =====================

    # Get the clustering models configuration from the config file
    models_config = get_models_config(config)
    k_range = range(*config["config_clustering"]["k_range"])

    # ------------------ Load datasets ------------------------
    path_datasets = log_data['log_data']['kept_datasets']

    path_exp = []
    for d in path_datasets:

        print(f" =============== DATASET {d} =============== ")
        data, labels = load_data_labels(d)
        clustering_true = get_clustering(labels)

        data, ts_dist = is_time_series(data)

        for model_name, model_config in models_config["config_clustering"].items():

            print(f" ---------------- MODEL {model_name} ---------------- ")
            t_start_exp = time.time()

            # ----------- Prepare clustering log --------------

            # We don't use the full path to the dataset only the path relative
            # to path_data
            d_shortname = d.replace(path_data, "")
            log_exp_fname = f"{path_res}{model_name}/{d_shortname}-clustering.json"
            log_exp = {
                "dataset": d,
                "k_true": len(np.unique(labels)),
                "model_name": model_name,
                "main_log_fname": log_fname,
                "log_fname" : log_exp_fname,
                "ts_dist": ts_dist,
                model_name: model_config,  # Add the config of this model
            }
            # Add current experiment log filename to the main log file
            path_exp.append(log_exp_fname)

            # ----------- Generate all clusterings --------------

            # Instanciate a scaler if one was provided in the config file
            if model_config['scaler'] is None:
                scaler = None
            else:
                scaler = model_config['scaler'](**model_config['scaler_kw'])

            # Generate all clusterings for the current dataset and model
            clusterings = generate_all_clusterings(
                data=data,
                model_class=model_config['model'],
                n_clusters_range=k_range,
                ts_dist=ts_dist,
                scaler=scaler,
                model_kw=model_config['model_kw'],
                fit_predict_kw=model_config['fit_predict_kw'],
                verbose=1,
            )

            # ----------- Compute VI and quality --------------
            VIs, qualities = compute_VI_quality(clustering_true, clusterings)


            # ----------- Finalize log and save --------------
            t_end_exp = time.time()
            dt = float(f"{t_end_exp - t_start_exp:.2f}")
            print(f"\n\nExperiment done in: {dt:.2f}s")


            log_exp["clusterings"] = clusterings
            log_exp["VIs"] = VIs
            log_exp["qualities"] = qualities
            log_exp["time"] = dt

            save_log(
                f"{log_exp_fname}", log_exp,
                overwrite=True, add_date=False, new_name=True, verbose=False,
            )

    # -------------------- Finalize log and save -----------------------
    t_end = time.time()
    dt = float(f"{t_end - t_start:.2f}")
    print(f"\n\nTotal execution time: {dt:.2f}s")

    log['log_clustering']["path_exp"] = path_exp
    log['log_clustering']["time"] = dt
    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )
    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log

def compute_CVI_values(config_fname:str, log_clustering_fname:Union[str, None] = None) -> dict:
    """
    Create CVI files for each non-filtered experiment in the log file

    - Relying on `log-clustering-20XX-XX-XX.json`.
    - Creates a text logfile `log-CVI-20XX-XX-XX.txt`. Fully reads `config-data.json` and `config-clustering.json`, `config-CVI.json` at the very beginning of the logfile.
    - Creates a json logfile `log-CVI-20XX-XX-XX.json` concatenating the config files used and giving some info about the general results
    - One key `config_data` with the corresponding dict
    - One key `log_data` with the corresponding dict
    - One key `config_clustering` with the corresponding dict
    - One key `log_clustering` with the corresponding dict
    - One key `config_CVI` with the provided config
    - One key `log_CVI` with the following keys:
        - `path_CVI` = list of all CVI json files created `[path/to/res/clustering_name/path/to/dataset-CVI.json]`
        - One key `kept_experiments` a list `"path/to/experiment/dataset-clustering.json"`, see `filter_experiments` function
        - One key `dropped_experiments` a dict `contraint : "path/to/experiment/dataset-clustering.json"`, see `filter_experiments` function
        - One key `kept_datasets` a list `[path/to/dataset]` for which at least one clustering method was kept
        - One key `dropped_datasets` a list `[path/to/dataset]` for which no clustering method was kept
        - `log_fname` : `path/to/res/log-CVI-20XX-XX-XX` (without the `.txt` or `.json`)
        - `time`
    - Filter experiments based on the contraints defined in `config_CVI` (see `filter_experiments` function)
    - Compute the values for each kept experiment and each provided CVI
    - Creates a `path_res/clustering_method/dataset-CVI.json` file for each kept experiment
    - `dataset`
    - `model_name`
    - `k_true`
    - `main_log_fname` (CVI)
    - `log_filename` (-CVI)
    - `log_experiment`
        - `main_log_fname` (clustering)
        - `log_fname` (-clustering)
        - model_name (model config)
        - `ts_dist`
    - `CVI_names`
    - CVI_name
        - `CVI_values`
        - `k_selected`
        - `time`

    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = load_config_as_dict(config_fname)
    if "config_CVI" not in config:
        raise ValueError("The config file must contain a 'config_CVI' key.")

    if log_clustering_fname is None:
        log_clustering = create_clusterings(config_fname)
    else:
        log_clustering = interpret_saved_dict(log_clustering_fname)

    path_data = log_clustering['config_data']['path_data']
    path_res = log_clustering['config_data']['path_res']

    # ----------- Prepare current log files ---------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    log_fname = f'{path_res}log-CVI-{full_date}'
    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        **log_clustering,
        "config_CVI": config['config_CVI'],
        "log_CVI": {
            "log_fname": log_fname,
        }}

    # Make the log visible in the output file
    print_log(log)

    # ================ Find experiments ===================
    all_experiments = log_clustering['log_clustering']['path_exp']

    # ================ Filter experiments ===================
    constraints = config['config_CVI'].copy()
    constraints.pop('path_data', None)
    constraints.pop('path_res', None)

    filtered_exp, filtered_datasets = filter_experiments(all_experiments)

    # ================ Create CVI files =====================
    path_CVI_files = []



    # -------------------- Finalize log and save -----------------------
    t_end = time.time()
    dt = float(f"{t_end - t_start:.2f}")
    print(f"\n\nTotal execution time: {dt:.2f}s")

    log['log_CVI']["path_CVI_files"] = path_CVI_files
    log['log_CVI']["time"] = dt
    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )
    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log