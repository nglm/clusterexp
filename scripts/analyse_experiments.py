#!/usr/bin/env python3

import os
import sys
import numpy as np
import json
from datetime import date
from pycvi.cvi import CVIs
from math import exp

from clusterexp.utils import (
    get_list_datasets, URL_ROOT,
    get_list_exp
)

DATA_SOURCE = "artificial"
#DATA_SOURCE = "real-world"

RES_DIR = f'./res/{DATA_SOURCE}/'
PATH = f"{URL_ROOT}{DATA_SOURCE}/"
FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"
# Store information dataset per dataset
FNAME_ANAL_DATASETS = f"analysis_datasets-{DATA_SOURCE}"
# Store information score per score
FNAME_ANAL_SCORES = f"analysis_scores-{DATA_SOURCE}"

def f_quality(VI):
    return exp(-2*VI)

def main():
    """
    Analyse the experiments by making statistics on each dataset and
    each score.

    For each score find:
    - the number of successful datasets
    - the number of successful datasets weighted by the quality
    - the quality of the successful dataset
    - the max quality that could have been obtained with the given
    clusterings.

    For each dataset find the number of successful scores (weighted with
    the VI)
    - the number of successful scores
    - the number of successful scores weighted by the quality
    - the quality of the successful scores
    - the max quality that could have been obtained with the given
    clusterings.

    Note that if a dataset has few successes and/or has a very low
    accumulated weighted VI, then it means that this dataset was just
    too hard to cluster or their definition of cluster might be weird.

    A score that has an unusual high accumulated VI compared to its
    number of success means that it is good at finding relevant clusters
    in general, but not necessarily *the* pre-defined number of clusters
    """


    res_datasets = {}

    datasets = get_list_datasets(RES_DIR + FNAME_DATASET_EXPS)

    # ================= Analysis Datasets =================

    for d in datasets:

        # ---------------- Initialisation of dataset ------------------

        # Here we gather global information about a specific dataset
        # Notably to evaluate how difficult this dataset was and how
        # "sensible" the true clustering was compared to what is commonly
        # considered as a good clustering by the methods and the CVIs
        res_datasets[d] = {
            # Global values
            "acc" : 0,            # +1 if k_true=k_selected (indpdt of quality)
            "weighted_acc" : 0,   # +w if k_true=k_selected
            "k_true_quality" : 0, # +w using k_true (indpdt of k_selected)
            "quality" : 0,        # +w using k_selected (indpdt of k_true)
            "max_quality" : 0,    # +w using k that has max quality
            "exps" : {},          # One dict per experience
            "scores" : {},        # One dict per score
            "k_true" : None,      # k_true of this dataset
            "k_best" : None,      # k that minimises VI for this dataset
        }

        # Initialise the accumulators on this dataset for each score
        # Here we gather information on how good this CVI did on this
        # dataset, combining the info of all clustering methods
        for s in CVIs:
            for cvi_type in s.cvi_types:
                res_datasets[d]["scores"][str(score)] = {
                    "acc" : 0,
                    "weighted_acc" : 0,
                    "k_true_quality" : 0,
                    "quality" : 0,
                    "max_quality" : 0,
                    "VIs_selected" : [], # VI of k_selected with this score
                    "VIs_true" : [],     # VI of k_true with this score
                    "VIs_best" : [],     # best reachable VI
                    "success" : [],
                }

        # ----------- Going through all clustering methods -------------

        # For each dataset, find all experiments working on this dataset
        # but using different clustering methods (by filtering on the
        # filename)
        fnames = get_list_exp(dataset_name=d, res_dir=RES_DIR)

        for fname in fnames:

            with open(RES_DIR + fname + ".json") as f_json:
                exp = json.load(f_json)

            # Change None to np.inf, which results in quality=0
            VIS_no_None = {
                k: vi if vi is not None else np.inf
                for k, vi in exp["VIs"].items()
            }
            k_true = exp["k"]
            k_best = min(VIS_no_None, key=VIS_no_None.get)

            VI_true = VIS_no_None[k_true]
            VI_best = VIS_no_None[k_best]

            # ------ Initialisation of experience ------
            res_exp = {
                "acc" : 0,
                "weighted_acc" : 0,
                "VI_true" : VI_true,
                "VI_best" : VI_best,
            }

            res_datasets[d]["k_true"] = k_true
            res_datasets[d]["k_best"] = k_best

            k_true_quality = f_quality(VIS_no_None[k_true])
            max_quality = f_quality(min(VIS_no_None))

            res_datasets[d]["k_true_quality"] = k_true_quality
            res_datasets[d]["max_quality"] = max_quality

            # ------ Update of experience ------

            # Update weighted accuracy for each score of this experiment
            for s in CVIs:
                for cvi_type in s.cvi_types:
                    score = s(cvi_type=cvi_type)
                    d_score = res_datasets[d]["scores"][str(score)]

                    # Find selected VI of the selected k
                    k_selected = exp["CVIs"][str(score)]["selected"]
                    VI_selected = VIS_no_None[k_selected]

                    d_score["VIs_selected"].append(VI_selected)
                    d_score["VIs_true"].append(VI_true)
                    quality_selected = f_quality(VI_selected)

                    # Compute weighted accuracy for this score
                    success = k_selected == k_true
                    d_score["success"].append(success)
                    if success:
                        d_score["acc"] += 1
                        res_exp["acc"] += 1
                        d_score["weighted_acc"] += quality_selected
                        res_exp["weighted_acc"] += quality_selected

                    d_score["quality"] += quality_selected
                    res_exp["quality"] += quality_selected

                    # True and max quality that we could obtain with this
                    # clustering algorithm
                    d_score["k_true_quality"] += k_true_quality
                    d_score["max_quality"] += max_quality

            # ------ Update of dataset ------

            # Update weighted accuracy for this dataset
            res_datasets[d]["acc"] += res_exp["acc"]
            res_datasets[d]["weighted_acc"] += res_exp["weighted_acc"]
            res_datasets[d]["quality"] += res_exp["quality"]
            res_datasets[d]["exps"][fname] = res_exp

    # ================= Analysis Scores =================

    res_scores = {
        "max_acc" : 0,        # +w using k_true with each experiment
        "scores" : {},
    }

    # ---------------- Initialisation of scores ------------------

    # Initialise the accumulators for each score
    for s in CVIs:
        for cvi_type in s.cvi_types:
            res_scores["scores"][str(score)] = {
                # +1 for each success (regardless of quality)
                "acc" : 0,
                # +w (regardless of success) using k_selected
                "weighted_acc" : 0,
            }

    for d in datasets:

        # Update res_scores using res_datasets
        res_scores["max_acc"] += res_datasets[d]["max_acc"]

        # Update accumulators of res_scores after going through all
        # experiments on this dataset
        for s in CVIs:
            for cvi_type in s.cvi_types:
                score = s(cvi_type=cvi_type)

                # Shorter variable names for dicts
                d = res_scores["scores"][str(score)]
                d_score = res_datasets[d]["scores"][str(score)]

                # Update res_scores using res_datasets
                d["acc"] += d_score["acc"]
                d["weighted_acc"] += d_score["weighted_acc"]

    # Save the analysis dict
    json_str = json.dumps(res_datasets, indent=2)
    with open(RES_DIR+FNAME_ANAL_DATASETS+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)
    json_str = json.dumps(res_scores, indent=2)
    with open(RES_DIR+FNAME_ANAL_SCORES+".json", 'w', encoding='utf-8') as f:
        f.write(json_str)

if __name__ == "__main__":
    main()