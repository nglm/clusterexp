#!/usr/bin/env python3

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

from pycvi.cluster import generate_all_clusterings
from pycvi.scores import SCORES
from pycvi.compute_scores import compute_all_scores
from pycvi.vi import variational_information

from clusterexp.utils import (
    load_data_from_github, get_data_labels, get_list_datasets,
    get_list_exp, URL_ROOT, load_json
)

import warnings
warnings.filterwarnings("ignore")

DATA_SOURCE = "artificial"
DATA_SOURCE = "real-world"

RES_DIR = f'./res/{DATA_SOURCE}/'
PATH = f"{URL_ROOT}{DATA_SOURCE}/"
FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"

SEED = 221


N_SCORES = 0
for s in SCORES:
    for score_type in s.score_types:
        N_SCORES += 1

N_ROWS = ceil(N_SCORES / 5)
N_COLS = 5
FIGSIZE = (4*N_COLS, ceil(2.5*N_ROWS))

def plot_clusters(data, clusterings, fig, titles):
    # Some datasets are in 3D
    (N, d) = data.shape
    # Plot the clustering selected by a given score
    for i in range(len(clusterings)):
        # Plot clusters one by one
        if d <= 2:
            ax = fig.axes[i+2] # i+2 because there are 2 plots already
        elif d == 3:
            ax = fig.add_subplot(N_ROWS, N_COLS, i+3, projection='3d')
        else:
            return None
        for i_label, cluster in enumerate(clusterings[i]):
            if d == 1:
                ax.scatter(np.zeros_like(data[cluster, 0]), data[cluster, 0], s=1)
            elif d == 2:
                ax.scatter(data[cluster, 0], data[cluster, 1], s=1)
            elif d == 3:
                ax.scatter(
                    data[cluster, 0], data[cluster, 1], data[cluster, 2], s=1
                )
        ax.set_title(str(titles[i]))
    return fig

def plot_true(data, labels, clusterings):
    # Some datasets are in 3D
    (N, d) = data.shape

    # ----------------------- Create figure ----------------
    if d <= 2:
        fig, axes = plt.subplots(
            nrows=N_ROWS, ncols=N_COLS, sharex=True, sharey=True, figsize=(20,10),
            tight_layout=True
        )
    elif d == 3:
        fig = plt.figure(figsize=(20,10), tight_layout=True)
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
        [labels == classes[i] for i in range(n_labels)],
        clusterings[n_labels]
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

        # Plot clusters one by one
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
        ax.set_title(ax_titles[i_ax])

    return fig

def compute_scores_VI(
    X, y, true_clusters,
    exp
):
    # all clusterings of this experiment
    clusterings = exp['clusterings']
    k_true = exp["k"]
    # best clusters
    best_clusters = clusterings[k_true]

    # ---------------- Plot true clusters ------------------------------
    ax_titles = []
    fig = plot_true(X, y, best_clusters)

    # ------------------------ Compute VIs ------------------------------
    # Compute VI between the true clustering and each clustering
    # obtained with the different clustering methods with the
    # real number of clusters
    VIs = {}

    print(" === VI === ")
    for k, clustering in clusterings.items():
        if clusterings is None:
            VIs[k] = None
        else:
            VIs[k] = variational_information(true_clusters, clustering)
        print(k, VIs[k])

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

            scores = compute_all_scores(
                score,
                X,
                [exp["clusterings"]],
                DTW=False,
                scaler=StandardScaler(),
            )

            k_selected = score.select(scores)[0]
            clusterings_selected.append(exp["clusterings"][k_selected])
            t_end = time.time()
            dt = t_end - t_start

            # Print and store score information
            for k in exp["clusterings"]:
                print(k, scores[0][k])
            print("Selected k {}".format(k_selected))
            print('Code executed in %.2f s' %(dt))

            exp["CVIs"][str(score)] = {
                "scores" : scores[0],
                "selected" : k_selected,
                "time" : dt,
            }

            ax_titles.append("{}, k={}, VI={}".format(
                score, k_selected, exp["VIs"][k_selected]))

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
    out_fname = './output-scores_VIs-' + today + ".txt"

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # --------- Get datasets and labels -----------
    datasets = get_list_datasets(RES_DIR + FNAME_DATASET_EXPS)
    # For each dataset, find all experiments working on this dataset
    # but using different clustering methods (by filtering on the
    # filename)
    t_start = time.time()
    for d in datasets:
        print(" ---------------- DATASET {} ---------------- ".format(d))

        fnames = get_list_exp(dataset_name=d, res_dir=RES_DIR)

        X, y, n_labels, meta = get_data_labels(d, path=PATH)
        classes = np.unique(y)

        # true clusters
        # List[List[int]]
        indices = np.arange(len(X))
        true_clusters = [
            indices[y == classes[i]] for i in range(n_labels)
        ]

        for fname in fnames:
            exp = load_json(f"{RES_DIR}{fname}.json")

            model_name = exp["model"]
            k_true = exp["k"]

            exp, fig = compute_scores_VI(X, y, true_clusters, exp)

            # ----- save figures and experiments -----
            exp_fname = f"{RES_DIR}{model_name}/{d}"

            # Save figure if it exists (d<=3)
            if fig is not None:
                figtitle = f"{d} - {model_name} - True k={k_true}"
                fig.suptitle(figtitle)
                fig.savefig(exp_fname + ".png")

            # Save experiment information as json
            json_str = json.dumps(exp, indent=2)
            with open(exp_fname + ".json", 'w', encoding='utf-8') as f:
                f.write(json_str)

    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}")
    fout.close()

if __name__ == "__main__":
    main()


