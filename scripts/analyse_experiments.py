#!/usr/bin/env python3

import os
import sys
import numpy as np
import json
import time
from datetime import date
from pycvi.cvi import CVIs
from math import exp
import argparse
from multiprocessing import Process

from clusterexp.utils import (
    get_list_datasets, get_list_exp,
    URL_ROOT, PATH_UCR_LOCAL, PATH_UCR_REMOTE
)

def define_globals(source_number: int = 0, local=True):
    """
    Dirty way of coding in order to parallelize the processes.

    :param source_number: index in ["artificial", "real-world", "UCR"]
    :type source_number: int, optional
    :param local: to know whether the local path should be used or not,
        defaults to True
    :type local: bool, optional
    """
    global DATA_SOURCE, RES_DIR, PATH
    global FNAME_DATASET_EXPS, FNAME_ANAL_DATASETS, FNAME_ANAL_CVIS
    global SEED, PATH_UCR

    sources = ["artificial", "real-world", "UCR"]
    DATA_SOURCE = sources[source_number]

    RES_DIR = f'./res/{DATA_SOURCE}/'

    PATH = f"{URL_ROOT}{DATA_SOURCE}/"
    FNAME_DATASET_EXPS = f"datasets_experiments_theory-{DATA_SOURCE}.txt"

    # Store information dataset per dataset
    FNAME_ANAL_DATASETS = f"analysis_datasets-{DATA_SOURCE}"
    # Store information CVI per CVI
    FNAME_ANAL_CVIS = f"analysis_CVIs-{DATA_SOURCE}"

    SEED = 221

    if local:
        PATH_UCR = PATH_UCR_LOCAL
    else:
        PATH_UCR = PATH_UCR_REMOTE


def f_quality(VI: float) -> float:
    """
    Quality of a clustering.

    The quality of a predicted clustering is based on the VI between the
    true and predicted clusterings. We define the quality as:

    :math:`quality = exp(-2*VI)`

    Which means that the quality is between 0 and 1, with higher quality
    values meaning better clusterings.

    Parameters
    ----------
    VI : float
        The Variation of Information between the true clustering and the
        predicted clustering

    Returns
    -------
    float
        The quality of the predicted clustering.
    """
    return exp(-2*VI)

def main():
    """
    Analyse the experiments by making statistics on each dataset and
    each CVI.

    For each CVI find:
    - `acc`: The number of successful datasets
    - `weighted_acc`: The number of successful datasets weighted by the
    quality.
    - `quality`: The accumulated quality of clusterings selected

    For each dataset find:
    - `acc`: The number of successful CVIs, using all clustering methods
    - `weighted_acc`: The number of successful CVIs weighted by the
    quality, using all clustering methods
    - `k_true_quality`: The accumulated quality of the clustering
    assuming k_true clusters, using all clustering methods. This is
    independent of CVIs.
    - `quality`: The accumulated quality of clusterings selected by each
    CVI, using all clustering methods. (Averaged over the number of
    CVIs)
    - `max_quality`: The accumulated max quality that could have been
    obtained with the given clusterings, using all clustering methods.
    This is independent of CVIs.
    - `CVIs`: A summary of the performance of each CVI, using all
    clustering methods (acc, weighted_acc, k_true_quality, quality,
    max_quality, VIs_selected, VIs_true, VIs_best, success)
    - `exps`: A summary of the performance of each clustering method,
    using all CVIs (acc, weighted_acc, quality, k_true_quality,
    max_quality, VI_true, VI_best)
    - `k_true`: The true number of clusters
    - `k_best`: The number of clusters that minimises VI for each
    clustering method

    Note that if a dataset has few successes and/or has a very low
    accumulated quality, then it means that this dataset was just
    too hard to cluster or their definition of cluster might be weird.

    A CVI that has an unusual high accumulated quality compared to its
    number of success means that it is good at finding relevant clusters
    in general, but not necessarily *the* pre-defined number of clusters
    """

    np.random.seed(SEED)
    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = f'./output-analyse_exp-{today}_{DATA_SOURCE}.txt'

    fout = open(out_fname, 'wt')
    sys.stdout = fout
    t_start = time.time()

    res_datasets = {}

    datasets = get_list_datasets(RES_DIR + FNAME_DATASET_EXPS)

    # ================= Analysis Datasets =================

    for d in datasets:
        print(f" ===== DATASET {d} =====", flush=True)

        # ---------------- Initialisation of dataset ------------------

        # Here we gather global information about a specific dataset
        # Notably to evaluate how difficult this dataset was and how
        # "sensible" the true clustering was compared to what is commonly
        # considered as a good clustering by the methods and the CVIs
        res_datasets[d] = {
            # Global values (for all CVIs and clustering methods)
            "acc" : 0,            # +1 if k_true=k_selected (indpdt of quality)
            "weighted_acc" : 0,   # +w if k_true=k_selected
            "k_true_quality" : 0, # +w using k_true (indpdt of k_selected)
            "quality" : 0,        # +w using k_selected (indpdt of k_true)
            "max_quality" : 0,    # +w using k of max quality (indpt of CVI)
            "exps" : {},          # One dict per experience
            "CVIs" : {},          # One dict per CVI
            "k_true" : None,      # k_true of this dataset
            "k_best" : [],        # k that minimises VI per clustering method
            "success" : False,    # True if <=1 CVI-clustering method worked
        }

        # Initialise for each CVI accumulators on this dataset (but for
        # all clustering methods)
        # Here we gather information on how good this CVI did on this
        # dataset, combining the info of all clustering methods
        for cvi in CVIs:
            for cvi_type in cvi.cvi_types:
                cvi_instance = cvi(cvi_type=cvi_type)
                res_datasets[d]["CVIs"][str(cvi_instance)] = {
                    # Accumulators, shared for all clustering methods
                    "acc" : 0,
                    "weighted_acc" : 0,
                    "k_true_quality" : 0,
                    "quality" : 0,
                    "max_quality" : 0,
                    # Lists, with one element per clustering method:
                    "VIs_selected" : [], # VI of k_selected with this CVI
                    "VIs_true" : [],     # VI of k_true
                    "VIs_best" : [],     # best reachable VI
                    "success" : [],      # List of booleans
                }

        # ----------- Going through all clustering methods -------------

        # For each dataset, find all experiments working on this dataset
        # but using different clustering methods (by filtering on the
        # filename)
        fnames = get_list_exp(
            dataset_name=d, res_dir=RES_DIR, suffix="_scored.json"
        )

        for fname in fnames:
            print(f"processing {fname}...", flush=True)

            with open(RES_DIR + fname + ".json") as f_json:
                exp = json.load(f_json)

            # VI of each assumption k for this clustering method
            # Change None to np.inf, which results in quality=0
            VIS_no_None = {
                k: (vi if vi is not None else np.inf)
                for k, vi in exp["VIs"].items()
            }
            VIS_no_None.pop("0", None)
            k_true = str(exp["k"])
            k_best = min(VIS_no_None, key=VIS_no_None.get)

            VI_true = VIS_no_None[k_true]
            VI_best = VIS_no_None[k_best]

            # ------ Initialisation of experience ------
            res_exp = {
                # Accumulators, shared for all CVIs on this clustering method
                "acc" : 0,
                "weighted_acc" : 0,
                "quality" : 0,
                # Single values (because all CVIs share the same value)
                "k_true_quality" : 0,
                "max_quality" : 0,
                "VI_true" : VI_true,
                "VI_best" : VI_best,
            }

            res_datasets[d]["k_true"] = k_true
            res_datasets[d]["k_best"].append(k_best)

            k_true_quality = f_quality(VIS_no_None[k_true])
            max_quality = f_quality(VIS_no_None[k_best])

            # Update dataset accumulators
            res_datasets[d]["k_true_quality"] += k_true_quality
            res_datasets[d]["max_quality"] += max_quality

            # Update clustering method single values
            res_exp["k_true_quality"] = k_true_quality
            res_exp["max_quality"] = max_quality

            print(f"k_true: {k_true}  |  k_best: {k_best}", flush=True)

            # ------ Update of experience ------

            # Update weighted accuracy for each CVI of this experiment
            cvi_done = {}
            for cvi in CVIs:
                for cvi_type in cvi.cvi_types:
                    cvi_instance = cvi(cvi_type=cvi_type)

                    # Ugly fix for ScoreFunction and CH_original for which
                    # "original" = absolute after initialisation......
                    if str(cvi_instance) in cvi_done:
                        continue
                    else:
                        cvi_done[str(cvi_instance)] = True

                    d_cvi = res_datasets[d]["CVIs"][str(cvi_instance)]

                    # Find selected VI of the selected k
                    k_selected = str(exp["CVIs"][str(cvi_instance)]["selected"])
                    if (k_selected is None) or (k_selected == "None"):
                        VI_selected = np.inf
                        success = False
                    else:
                        VI_selected = VIS_no_None[k_selected]
                        success = k_selected == k_true

                    d_cvi["VIs_selected"].append(VI_selected)
                    d_cvi["VIs_true"].append(VI_true)
                    d_cvi["VIs_best"].append(VI_best)
                    quality_selected = f_quality(VI_selected)

                    # Compute weighted accuracy for this CVI
                    d_cvi["success"].append(success)
                    if success:
                        res_datasets[d]["success"] = True
                        d_cvi["acc"] += 1
                        res_exp["acc"] += 1
                        d_cvi["weighted_acc"] += quality_selected
                        res_exp["weighted_acc"] += quality_selected

                    d_cvi["quality"] += quality_selected
                    res_exp["quality"] += quality_selected

                    # True and max quality that we could obtain with this
                    # clustering algorithm
                    d_cvi["k_true_quality"] += k_true_quality
                    d_cvi["max_quality"] += max_quality

                    res_datasets[d]["CVIs"][str(cvi_instance)] = d_cvi

            # ------ Update of dataset ------

            # Average quality over the number of CVIs
            res_exp["quality"] /= len(res_datasets[d]["CVIs"])
            # Updates
            res_datasets[d]["acc"] += res_exp["acc"]
            res_datasets[d]["weighted_acc"] += res_exp["weighted_acc"]
            res_datasets[d]["quality"] += res_exp["quality"]
            res_datasets[d]["exps"][fname] = res_exp

            msg = (
                f'acc: {res_exp["acc"]}  |  ' +
                f'weighted_acc: {res_exp["weighted_acc"]}  |  ' +
                f'quality: {res_exp["quality"]}  |  '
            )

            print(msg, flush=True)

    # ================= Analysis CVIs =================

    res_CVIs = {
        # Global:
        "max_quality" : 0,  # +w using k that has max quality (indpt of CVI)
        # CVI dependant:
        "CVIs" : {},        # One dict per CVI
    }

    # ---------------- Initialisation of CVIs ------------------

    # Initialise the accumulators for each CVI
    for cvi in CVIs:
        for cvi_type in cvi.cvi_types:
            cvi_instance = cvi(cvi_type=cvi_type)
            res_CVIs["CVIs"][str(cvi_instance)] = {
                "acc" : 0,          # +1 for each success (indpdt of quality)
                "weighted_acc" : 0, # +w for each success (ie k_true=k_selected)
                "quality" : 0,      # +w using k_selected (indpdt of k_true)

            }

    # ------------ Update all CVIs using res_datasets ----------
    for d in datasets:

        # Update res_CVIs using res_datasets
        res_CVIs["max_quality"] += res_datasets[d]["max_quality"]

        # Update accumulators of res_CVIs after going through all
        # experiments on this dataset
        cvi_done = {}
        for cvi in CVIs:
            for cvi_type in cvi.cvi_types:
                cvi_instance = cvi(cvi_type=cvi_type)

                # Ugly fix for ScoreFunction and CH_original for which
                # "original" = absolute after initialisation......
                if str(cvi_instance) in cvi_done:
                    continue
                else:
                    cvi_done[str(cvi_instance)] = True

                # Shorter variable names for dicts
                res_cvi = res_CVIs["CVIs"][str(cvi_instance)]
                d_cvi = res_datasets[d]["CVIs"][str(cvi_instance)]

                # Update res_CVIs using res_datasets
                res_cvi["acc"] += d_cvi["acc"]
                res_cvi["weighted_acc"] += d_cvi["weighted_acc"]
                res_cvi["quality"] += d_cvi["quality"]

                res_CVIs["CVIs"][str(cvi_instance)] = res_cvi

    # --------------- Print out CVI results -----------------
    cvi_done = {}
    for cvi in CVIs:
        for cvi_type in cvi.cvi_types:
            cvi_instance = cvi(cvi_type=cvi_type)

            # Ugly fix for ScoreFunction and CH_original for which
            # "original" = absolute after initialisation......
            if str(cvi_instance) in cvi_done:
                continue
            else:
                cvi_done[str(cvi_instance)] = True

            d = res_CVIs["CVIs"][str(cvi_instance)]

            print(f" ===== CVI {str(cvi_instance)} =====", flush=True)
            msg = (
                f'acc: {d["acc"]}  |  ' +
                f'weighted_acc: {d["weighted_acc"]}  |  ' +
                f'quality: {d["quality"]}  |  '
            )
            print(msg, flush=True)


    # ================= Saving dictionaries =================
    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}")
    fout.close()

    # Save the analysis dict
    json_str = json.dumps(res_datasets, indent=2)
    with open(RES_DIR+FNAME_ANAL_DATASETS+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)
    json_str = json.dumps(res_CVIs, indent=2)
    with open(RES_DIR+FNAME_ANAL_CVIS+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)

def run_process(source_number, local):
    define_globals(source_number, local)
    main()

if __name__ == "__main__":
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
        "--local", # Drop `--` for positional/required params
        nargs=1,  # creates a list of one element
        type=int,
        default=0,  # default if nothing is provided
    )
    CLI.add_argument(
        "--source_num", # Drop `--` for positional/required params
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default=[0, 1, 2],  # default if nothing is provided
    )
    args = CLI.parse_args()

    local = bool(int(args.local[0]))
    source_numbers = args.source_num

    processes = [
        Process(target=run_process, args=(i, local))
        for i in source_numbers
    ]

    # kick them off
    for process in processes:
        process.start()
    # now wait for them to finish
    for process in processes:
        process.join()
