#!/usr/bin/env python3

from multiprocessing import Process
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from datetime import date
import sys
import json
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor
from typing import List
import os
import gc

from pycvi.cluster import generate_all_clusterings
from pycvi.scores import SCORES
from pycvi.compute_scores import compute_all_scores
from pycvi.vi import variational_information

from clusterexp.utils import (
    get_data_labels, get_list_datasets,
    get_list_exp, URL_ROOT, load_json, get_fname, get_data_labels_UCR,
    URL_ROOT, PATH_UCR_LOCAL, PATH_UCR_REMOTE
)

import warnings
warnings.filterwarnings("ignore")

# --------  Adapt the figures to the total number of scores ------------
N_SCORES = 0
for s in SCORES:
    for score_type in s.score_types:
        N_SCORES += 1

N_ROWS = ceil(N_SCORES / 5)
N_COLS = 5
FIGSIZE = (4*N_COLS, ceil(2.5*N_ROWS))
# ----------------------------------------------------------------------

def define_globals(source_number: int = 0, local=True):
    """
    Dirty way of coding in order to parallelize the processes.

    :param source_number: index in ["artificial", "real-world", "UCR"]
    :type source_number: int, optional
    :param local: to know whether the local path should be used or not,
        defaults to True
    :type local: bool, optional
    """
    global DATA_SOURCE, DTW
    global RES_DIR, PATH, FNAME_DATASET_EXPS
    global SEED, PATH_UCR

    sources = ["artificial", "real-world", "UCR"]
    DATA_SOURCE = sources[source_number]

    RES_DIR = f'./res/{DATA_SOURCE}/'

    PATH = f"{URL_ROOT}{DATA_SOURCE}/"
    FNAME_DATASET_EXPS = f"datasets_experiments_theory-{DATA_SOURCE}.txt"

    SEED = 221
    DTW = None

    if local:
        PATH_UCR = PATH_UCR_LOCAL
    else:
        PATH_UCR = PATH_UCR_REMOTE


def plot_clusters(
    data: np.ndarray,
    clusterings_selected: List[List[List[int]]],
    fig,
    titles: List[str],
):
    """
    Add one plot per score with their corresponding selected clustering

    The fig should already contain 2 plots first with the true
    clusterings, and the clusterings obtained with k_true.

    :param data: Original data, corresponding to a benchmark dataset
    :type data: np.ndarray, shape (N, d)
    :param clusterings_selected: A list of n_score clusterings.
    :type clusterings_selected: List[List[List[int]]]
    :param fig: Figure where all the plots are (including 2 about the
        true clusters)
    :type fig:
    :param titles: List of titles for each score
    :type titles: List[str]
    :return: a figure with one clustering per score (+2 plots first)
    :rtype:
    """
    (N, d) = data.shape
    # -------  Plot the clustering selected by a given score -----------
    for i_score in range(len(clusterings_selected)):

        # ------------- Find the ax corresponding to the score ---------
        if d <= 2:
            ax = fig.axes[i_score+2] # i+2 because there are 2 plots already
        # Some datasets are in 3D
        elif d == 3:
            ax = fig.add_subplot(N_ROWS, N_COLS, i_score+3, projection='3d')
        else:
            return None

        # Add predefined title
        ax.set_title(str(titles[i_score]))

        # Sometimes there are no clusters selected because all scores were None
        if clusterings_selected[i_score] is None:
            continue

        # ------------------ Plot clusters one by one ------------------
        for i_label, cluster in enumerate(clusterings_selected[i_score]):
            if d == 1:
                ax.scatter(np.zeros_like(data[cluster, 0]), data[cluster, 0], s=1)
            elif d == 2:
                ax.scatter(data[cluster, 0], data[cluster, 1], s=1)
            elif d == 3:
                ax.scatter(
                    data[cluster, 0], data[cluster, 1], data[cluster, 2], s=1
                )

    return fig

def plot_true(
    data: np.ndarray,
    labels: np.ndarray,
    clusterings: List[List[List[int]]],
):
    """
    Plot the true clustering and the clustering obtained with k_true

    Create also the whole figure that will be used to plot the
    clusterings selected by each score.

    :param data: Original data, corresponding to a benchmark dataset
    :type data: np.ndarray, shape (N, d)
    :param labels: True labels
    :type labels: np.ndarray
    :param clusterings: The clusterings obtained with k_true
    :type clusterings: List[List[List[int]]]
    :return: _description_
    :rtype: _type_
    """

    (N, d) = data.shape

    # ----------------------- Create figure ----------------
    if d <= 2:
        fig, axes = plt.subplots(
            nrows=N_ROWS, ncols=N_COLS, sharex=True, sharey=True,
            figsize=FIGSIZE, tight_layout=True
        )
    # Some datasets are in 3D
    elif d == 3:
        fig = plt.figure(figsize=FIGSIZE, tight_layout=True)
    else:
        return None

    # ----------------------- Labels ----------------
    if labels is None:
        labels = np.zeros(N)
    classes = np.unique(labels)
    n_labels = len(classes)
    if n_labels == N:
        labels = np.zeros(N)
        n_labels = 1

    # ------------------- variables for the 2 axes ----------------
    clusters = [
        # The true clustering
        [labels == classes[i] for i in range(n_labels)],
        # The clustering obtained with k_true
        clusterings
    ]
    ax_titles = [
        "True labels, k={}".format(n_labels),
        "Clustering assuming k={}".format(n_labels),
    ]

    # ------ True clustering and clustering assuming n_labels ----------
    for i_ax in range(2):
        if d <= 2:
            ax = fig.axes[i_ax]
        elif d == 3:
            ax = fig.add_subplot(N_ROWS, N_COLS, i_ax+1, projection='3d')

        # ---------------  Plot clusters one by one --------------------
        for i in range(n_labels):

            c = clusters[i_ax][i]
            if d == 1:
                ax.scatter(
                    np.zeros_like(data[c, 0]), data[c, 0], s=1
                )
            elif d == 2:
                ax.scatter(data[c, 0], data[c, 1], s=1)
            elif d == 3:
                ax.scatter(
                    data[c, 0], data[c, 1], data[c, 2], s=1
                )

        # Add title
        ax.set_title(ax_titles[i_ax])

    return fig

def compute_scores_VI(
    X: np.ndarray,
    y: np.ndarray,
    true_clusters: List[List[int]],
    exp: dict,
):
    """
    Compute scores, VI and return a summary figure with updated exp

    :param X: Original data, corresponding to a benchmark dataset
    :type X: np.ndarray, shape (N, d)
    :param y: True labels
    :type y: np.ndarray
    :param true_clusters: The original clustering
    :type true_clusters: List[List[int]]
    :param exp: The dictionary summarizing the experiment
    :type exp: dict
    :return: The updated experiment (with scores and VI) and a figure
        with the true clustering, the clustering obtained with k_true
        and the selected clustering for each score
    :rtype:
    """
    # all clusterings of this experiment
    clusterings = exp['clusterings']
    k_true = exp["k"]
    # best clusters
    best_clusters = clusterings[k_true]

    # ---------------- Plot true clusters ------------------------------
    ax_titles = []
    if DATA_SOURCE == "artificial":
        fig = plot_true(X, y, best_clusters)
    else:
        fig = None

    # ------------------------ Compute VIs ------------------------------
    # Compute VI between the true clustering and each clustering
    # obtained with the different clustering methods with the
    # real number of clusters
    VIs = {}

    print(" === VI === ", flush=True)
    for k, clustering in clusterings.items():
        if clustering is None:
            VIs[k] = None
        else:
            VIs[k] = variational_information(true_clusters, clustering)
        print(k, VIs[k], flush=True)

    # Add VI to the exp file dict
    exp["VIs"] = VIs

    # ------------------------ Compute scores --------------------------
    clusterings_selected = []
    exp["CVIs"] = {}
    for s in SCORES:
        for score_type in s.score_types:
            score = s(score_type=score_type)
            print(" ================ {} ================ ".format(score))
            t_start = time.time()

            # Dirty workaround because old files don't have the DTW key
            if "DTW" in exp:
                DTW = bool(exp["DTW"])
            else:
                DTW = "_True/" in exp["fname"]
                exp["DTW"] = DTW

            scores = compute_all_scores(
                score,
                X,
                [exp["clusterings"]],
                DTW= DTW,
                scaler=StandardScaler(),
            )

            # Sometimes there is no k_selected because all scores were None
            k_selected = score.select(scores)[0]
            if k_selected is None:
                clusterings_selected.append(None)
            else:
                clusterings_selected.append(exp["clusterings"][k_selected])
            t_end = time.time()
            dt = t_end - t_start

            # Print and store score information
            for k in exp["clusterings"]:
                print(k, scores[0][k], flush=True)
            print(f"Selected k: {k_selected} | True k: {k_true}", flush=True)
            print('Code executed in %.2f s' %(dt))

            exp["CVIs"][str(score)] = {
                "scores" : scores[0],
                "selected" : k_selected,
                "time" : dt,
            }

            if k_selected is None:
                ax_titles.append(
                    f"{score}, No k could be selected"
                )
            else:
                ax_titles.append(
                    f"{score}, k={k_selected}, VI={exp['VIs'][k_selected]:.4f}"
                )

    # ---------------- Plot selected clusterings -----------------------
    if fig is not None:
        fig = plot_clusters(X, clusterings_selected, fig, ax_titles)

    return exp, fig

def main():
    """
    Compute and store scores and figures and scores for all experiments
    """

    np.random.seed(SEED)

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = f'./output-scores_VIs-{today}_{DATA_SOURCE}.txt'

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # --------- Get datasets and labels -----------
    datasets = get_list_datasets(RES_DIR + FNAME_DATASET_EXPS)
    # For each dataset, find all experiments working on this dataset
    # but using different clustering methods (by filtering on the
    # filename)
    t_start = time.time()
    for d in datasets:
        print(f" ---------------- DATASET {d} ---------------- ")

        fnames = get_list_exp(dataset_name=d, res_dir=RES_DIR)

        if DATA_SOURCE == "UCR":
            f = f"{PATH_UCR}{get_fname(d, data_source=DATA_SOURCE)}"
            X, y, n_labels, _ = get_data_labels_UCR(f, path=PATH)
        else:
            X, y, n_labels, meta = get_data_labels(d, path=PATH)
        classes = np.unique(y)
        print(f"k_true = {n_labels}", flush=True)

        # true clusters
        # List[List[int]]
        indices = np.arange(len(X))
        true_clusters = [
            indices[y == classes[i]] for i in range(n_labels)
        ]

        for fname in fnames:

            # ----- prepare figures and experiments files -----
            exp = load_json(f"{RES_DIR}{fname}.json")
            model_name = exp["model"]

            os.makedirs(f"{RES_DIR}{model_name}_{DTW}", exist_ok=True)
            exp_fname = f"{RES_DIR}{model_name}_{DTW}/{d}"
            exp_fname_json = f"{exp_fname}_scored.json"
            # Don't run again if the score file already exists
            if os.path.isfile(exp_fname_json):
                print(f"{exp_fname_json} already exists.")

            # ----------- Compute scores and VI ---------------
            else:

                exp, fig = compute_scores_VI(X, y, true_clusters, exp)

                # Save figure if it exists ("artificial" and d<=3)
                if fig is not None:
                    k_true = exp["k"]
                    figtitle = f"{d} - {model_name} - True k={k_true}"
                    fig.suptitle(figtitle)
                    fig.savefig(exp_fname + ".png")

                # Save experiment information as json
                json_str = json.dumps(exp, indent=2)
                with open(exp_fname_json, 'w', encoding='utf-8') as f:
                    f.write(json_str)

            # Try to limit memory usage...
            gc.collect()


    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}")
    fout.close()

def run_process(source_number, local):
    define_globals(source_number, local)
    main()

if __name__ == "__main__":
    source_numbers = range(3)
    local = bool(int(sys.argv[1]))

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

