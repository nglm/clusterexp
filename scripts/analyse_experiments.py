#!/usr/bin/env python3
"""
For each score find the number of successful datasets (weighted with the VI)
For each dataset find the number of successful scores (weighted with the VI)

Have the same pipeline with the UCR archive.
"""

import os
import sys
import numpy as np
import json
from datetime import date
from pycvi.scores import SCORES
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

def main():
    """
    Analyse the experiments by making statistics on each dataset and
    each score.

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

        res_datasets[d] = {
            # Global values
            "acc" : 0,            # +1 for each success (regardless of quality)
            "weighted_acc" : 0,   # +w (regardless of success) using k_selected
            "max_acc" : 0,        # +w using k_true with each experiment
            "exps" : {},          # One dict per experience
            "scores" : {},        # One dict per score
            "k_true" : None,
        }

        # For each dataset, find all experiments working on this dataset
        # but using different clustering methods (by filtering on the
        # filename)
        fnames = get_list_exp(dataset_name=d, res_dir=RES_DIR)

        # Initialise the accumulators on this dataset for each score
        for s in SCORES:
            for score_type in s.score_types:
                res_datasets[d]["scores"][str(score)] = {
                    "acc" : 0,
                    "weighted_acc" : 0,
                    "VIs_selected" : [], # VI of k_selected with this score
                    "success" : [],
                }

        for fname in fnames:

            # ------ Initialisation of experience ------

            with open(RES_DIR + fname + ".json") as f_json:
                exp = json.load(f_json)

            k_true = exp["k"]
            VI_true = exp["VIs"][str(k_true)]

            res_exp = {
                "acc" : 0,
                "weighted_acc" : 0,
                "VI_true" : VI_true,
            }

            res_datasets[d]["k_true"] = k_true

            # ------ Update of experience ------

            # Update weighted accuracy for each score of this experiment
            for s in SCORES:
                for score_type in s.score_types:
                    score = s(score_type=score_type)
                    d_score = res_datasets[d]["scores"][str(score)]

                    # Find selected VI of the selected k
                    k_selected = exp["CVIs"][str(score)]["selected"]
                    VI_selected = exp["VIs"][str(k_selected)]
                    d_score["VIs_selected"].append(VI_selected)

                    # Compute weighted accuracy for this score
                    success = k_selected == k_true
                    d_score["success"].append(success)
                    if success:
                        d_score["acc"] += 1
                        res_exp["acc"] += 1
                    d_score["weighted_acc"] += exp(-2*VI_selected)
                    res_exp["weighted_acc"] += exp(-2*VI_selected)

            # ------ Update of dataset ------

            # Update weighted accuracy for this dataset
            res_datasets[d]["acc"] += res_exp["acc"]
            res_datasets[d]["weighted_acc"] += res_exp["weighted_acc"]
            res_datasets[d]["max_acc"] += exp(-2*VI_true)
            res_datasets[d]["exps"][fname] = res_exp

    # ================= Analysis Scores =================

    res_scores = {
        "max_acc" : 0,        # +w using k_true with each experiment
        "scores" : {},
    }

    # ---------------- Initialisation of scores ------------------

    # Initialise the accumulators for each score
    for s in SCORES:
        for score_type in s.score_types:
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
        for s in SCORES:
            for score_type in s.score_types:
                score = s(score_type=score_type)

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