"""Experiment orchestration for full pipeline."""

import sys
from datetime import datetime
import numpy as np
import time
from pycvi.cluster import get_clustering, generate_all_clusterings
from pycvi.compute_scores import compute_all_scores
from pycvi.exceptions import SelectionError

from .config import interpret_config, get_models_config, get_mandatory_keys
from .data import find_datasets, filter_datasets, load_data_labels, is_time_series
from .utils import save_log, print_log, interpret_dict
from .clustering import (
    compute_VI_quality, filter_experiments, group_exp_by_dataset
)

from typing import Union

def prepare_data(config_fname:str) -> dict:
    """
    Prepare datasets and write a data-selection log.

    - Reads the config file: function `interpret_config`
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

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_data`` section.

    Returns
    -------
    dict
        Log dictionary containing the interpreted data config, dataset
        filtering results, and the generated log filename.
    """
    # ---------------- Read config file ---------------------
    config = interpret_config(config_fname)
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

    filtered_datasets = filter_datasets(
        datasets, path_data=path_data, **constraints
    )

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

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_clustering`` section.
    log_data_fname : Union[str, None], optional
        Path to an existing data log. When omitted, :func:`prepare_data`
        is called first.

    Returns
    -------
    dict
        Log dictionary combining data-selection information, clustering
        configuration, experiment file paths, and timing information.
    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = interpret_config(config_fname)
    if "config_clustering" not in config:
        raise ValueError("The config file must contain a 'config_clustering' key.")

    if log_data_fname is None:
        log_data = prepare_data(config_fname)
    else:
        log_data = interpret_dict(log_data_fname)

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

        # Load dataset, labels and true clustering
        data, labels = load_data_labels(f"{path_data}{d}")
        clustering_true = get_clustering(labels)
        k_true = len(np.unique(labels))

        print(f" ======== DATASET {d} | k_true = {k_true} ========== ")

        data, ts_dist = is_time_series(data)

        for model_name, model_config in models_config["config_clustering"].items():

            print(f" ---------------- MODEL {model_name} ---------------- ")
            t_start_exp = time.time()

            # ----------- Prepare clustering log --------------

            log_exp_fname = f"{path_res}{model_name}/{d}-clustering.json"

            log_exp = {
                "dataset": d,
                "k_true": k_true,
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
    Create CVI result files for each non-filtered clustering experiment.

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
        - `CVI_names` = list of CVI names used
        - `log_fname` : `path/to/res/log-CVI-20XX-XX-XX` (without the `.txt` or `.json`)
        - `time`
    - Filter experiments based on the contraints defined in `config_CVI` (see `filter_experiments` function)
    - Compute the values for each kept experiment and each provided CVI
    - Creates a `path_res/clustering_method/dataset-CVI.json` file for each kept experiment
    - `dataset`
    - `k_true`
    - `model_name`
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

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_CVI`` section.
    log_clustering_fname : Union[str, None], optional
        Path to an existing clustering log. When omitted,
        :func:`create_clusterings` is called first.

    Returns
    -------
    dict
        Log dictionary containing CVI configuration, filtered
        experiments, generated CVI file paths, and timing information.

    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = interpret_config(config_fname)
    if "config_CVI" not in config:
        raise ValueError("The config file must contain a 'config_CVI' key.")

    if log_clustering_fname is None:
        log_clustering = create_clusterings(config_fname)
    else:
        log_clustering = interpret_dict(log_clustering_fname)

    path_data = log_clustering['config_data']['path_data']
    path_res = log_clustering['config_data']['path_res']

    # Get the clustering models configuration from the config file
    models_config = get_models_config(config)
    CVI_names = list(models_config["config_CVI"].keys())

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
            "CVI_names": CVI_names,
        }}

    # Make the log visible in the output file
    print_log(log)

    # ================ Find experiments ===================
    all_experiments = log_clustering['log_clustering']['path_exp']

    # ================ Filter experiments ===================

    # Find constraints from config by finding mandatory keys != seed
    constraint_names = [
        k for k in get_mandatory_keys()['config_CVI']["mandatory"]
        if k != "seed"
    ]
    constraints = {
        c: config['config_CVI'][c] for c in constraint_names
        if c in config['config_CVI']
    }

    # Filter experiments and datasets
    filtered_exp, filtered_datasets = filter_experiments(
        all_experiments, **constraints
    )

    # Group experiments by dataset
    grouped_exp = group_exp_by_dataset(filtered_exp["kept_experiments"])

    # ================ Create CVI files =====================
    path_CVI_files = []

    # Note that we could avoid grouping by dataset but then it's bit less clean
    # and we would have to load the data and labels for each experiment
    for d, experiments in grouped_exp.items():

        data, labels = load_data_labels(f"{path_data}{d}")
        k_true = len(np.unique(labels))

        print(f" ======== DATASET {d} | k_true = {k_true} ========== ")

        data, ts_dist = is_time_series(data)

        # Prepare main log to store the selected k for each CVI and dataset
        log["log_CVI"][d] = {"k_true": k_true}

        for exp in experiments:

            # ----------- Read clustering log --------------

            # Read the clustering experiment log file
            log_exp = interpret_dict(exp)

            # Retrieve the model name from the log
            model_name = log_exp["model_name"]

            # Retrieve the clusterings from the log
            clusterings = log_exp["clusterings"]

            # Retrieve scaler
            scaler_class = log_exp[model_name]["scaler"]
            scaler_kw = log_exp[model_name]["scaler_kw"]
            if scaler_class is None:
                scaler = None
            else:
                scaler = scaler_class(**scaler_kw)

            # ----------- Prepare CVI log --------------
            log_cvi_fname = f"{path_res}{model_name}/{d}-CVI.json"

            log_cvi = {
                "dataset": d,
                "k_true": k_true,
                "model_name": model_name,
                "ts_dist": ts_dist,
                "main_log_fname": log_fname,
                "log_fname" : log_cvi_fname,
                "log_experiment": {
                    "main_log_fname": log_exp["main_log_fname"],
                    "log_fname": log_exp["log_fname"],
                    model_name: log_exp[model_name],
                },
                "CVI_names": CVI_names,
            }
            # Add current CVI log filename to the main log file
            path_CVI_files.append(log_cvi_fname)

            for cvi, cvi_config in models_config["config_CVI"].items():


                # ============ Compute CVI values =============
                print(f" ================ {cvi} ================ ")
                t_start_cvi = time.time()

                cvi_instance = cvi_config["cvi"](**cvi_config["cvi_init_kw"])

                cvi_values = compute_all_scores(
                    cvi_instance,
                    data,
                    clusterings,
                    ts_dist=ts_dist,
                    scaler=scaler,
                    rng=config["config_CVI"]["seed"],
                    cvi_kwargs=cvi_config["cvi_kw"],
                )

                # if all cvi values were None, no k selected
                try:
                    k_selected = cvi_instance.select(cvi_values)
                except SelectionError as e:
                    k_selected = None

                # ----------- Update logs --------------
                # Print cvi information
                for k, cvi_value in cvi_values.items():
                    print(k, cvi_value, flush=True)
                print(f"Selected k: {k_selected} | True k: {k_true}", flush=True)

                t_end_cvi = time.time()
                dt = t_end_cvi - t_start_cvi
                print('Code executed in %.2f s' %(dt))

                log_cvi[cvi] = {
                    "cvi_values" : cvi_values,
                    "selected" : k_selected,
                    "time" : dt,
                }

                # Update main log with the selected k for this CVI and dataset
                log['log_CVI'][d][cvi] = k_selected

            # ----------- Save experiment log ------------------
            save_log(
                f"{log_cvi_fname}", log_cvi,
                overwrite=True, add_date=False, new_name=True, verbose=False,
            )


    # -------------------- Finalize log and save -----------------------
    t_end = time.time()
    dt = float(f"{t_end - t_start:.2f}")
    print(f"\n\nTotal execution time: {dt:.2f}s")

    log['log_CVI'].update(filtered_exp)
    log['log_CVI'].update(filtered_datasets)
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