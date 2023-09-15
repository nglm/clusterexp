#!/usr/bin/env python3

# Courtesy to https://github.com/deric/clustering-benchmark

from scipy.io import arff
import numpy as np
import pandas as pd
import requests
import io
import urllib.request
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn_extra.robust import RobustWeightedKMeans
from datetime import date
import sys
import json
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil, floor

from pycvi.cluster import generate_all_clusterings
from pycvi.scores import Inertia, GapStatistic, ScoreFunction, Hartigan, Diameter, CalinskiHarabasz, Silhouette, SCORES
from pycvi.compute_scores import compute_all_scores
import warnings
warnings.filterwarnings("ignore")

URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"

# Just one cluster
UNIMODAL = [
    "birch-rg1.arff", "birch-rg2.arff",
    "golfball.arff",
]
# No class column in the data
UNLABELED = [
    "birch-rg1.arff", "birch-rg2.arff",
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set3.arff",
]

# Unknown number of clusters
UNKNOWN_K = [
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",
]

N_SCORES = 0
for s in SCORES:
    for score_type in s.score_types:
        N_SCORES += 1
N_ROWS = ceil(N_SCORES / 5)
N_COLS = 5
FIGSIZE = (4*N_COLS, ceil(2.5*N_ROWS))

def arff_from_github(url):
    ftpstream = urllib.request.urlopen(url)
    data, meta = arff.loadarff(io.StringIO(ftpstream.read().decode('utf-8')))
    return data, meta

def load_data_from_github(url, with_labels=True):
    data, meta = arff_from_github(url)
    df = pd.DataFrame(data)
    # Get only data, not the labels and convert to numpy
    if with_labels:
        data = df.iloc[:, 0:-1].to_numpy()
        labels = df.iloc[:, -1].to_numpy()
    else:
        data = df.to_numpy()
        labels = None
    return data, labels, meta

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

def experiment(
    X,
    y,
    model_class = AgglomerativeClustering,
    n_clusters_range = [i for i in range(25)],
    model_kw = {},
):

    exp = {}
    clusterings_selected = []
    ax_titles = []

    N = len(X)
    if N > 10000:
        print("Dataset too big {}".format(N))
    else:

        clusterings = generate_all_clusterings(
                X,
                model_class,
                n_clusters_range,
                DTW=False,
                scaler=StandardScaler(),
                model_kw=model_kw,
            )

        # store clustering information
        exp["clusterings"] = clusterings[0]
        fig = plot_true(X, y, clusterings[0])

        for s in SCORES:
            for score_type in s.score_types:
                score = s(score_type=score_type)
                print(" ================ {} ================ ".format(score))
                t_start = time.time()

                scores = compute_all_scores(
                    score,
                    X,
                    clusterings,
                    DTW=False,
                    scaler=StandardScaler(),
                )

                k_selected = score.select(scores)[0]
                clusterings_selected.append(clusterings[0][k_selected])
                t_end = time.time()
                dt = t_end - t_start

                # Print and store score information
                for k in n_clusters_range:
                    print(k, scores[0][k])
                print("Selected k {}".format(k_selected))
                print('Code executed in %.2f s' %(dt))

                exp[str(s)] = {
                    "scores" : scores[0],
                    "selected" : k_selected,
                    "time" : dt,
                }

                ax_titles.append("{}, k={}".format(score, k_selected))

    fig = plot_clusters(X, clusterings_selected, fig, ax_titles)

    return exp, fig

def main():

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = './output-' + today + ".txt"

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # --------- Get datasets and labels -----------
    fname_summary = URL_ROOT + "artificial.txt"
    raw_text = requests.get(fname_summary).text
    all_datasets = raw_text.split("\n")

    l_data = []
    l_labels = []
    l_n_labels = []
    l_fname = [f for f in all_datasets if f not in UNKNOWN_K]
    for fname in l_fname:
        if fname in UNLABELED:
            with_labels = False
            if fname in UNIMODAL:
                n_labels = 1
            else:
                n_labels = None
        else:
            with_labels = True
        data, labels, meta = load_data_from_github(
            PATH + fname, with_labels=with_labels
        )
        if with_labels:
            n_labels = len(np.unique(labels))
            if n_labels == len(data):
                n_labels = 1
        if len(data) <= 10000 and n_labels <= 20:
            l_data.append(data)
            l_labels.append(labels)
            l_n_labels.append(n_labels)
            print(
                "Dataset: {}  | Shape: {}  | #labels: {}".format(
                    fname, data.shape, n_labels
                )
            )

    # ------ Run experiments for all datasets and all scores -----------
    # model_classes = [
    #     AgglomerativeClustering, AgglomerativeClustering,
    #     SpectralClustering,
    #     ]
    # model_names = [
    #     "AgglomerativeClustering-Single", "AgglomerativeClustering-Ward",
    #     "SpectralClustering",
    # ]
    # model_kws = [
    #     {"linkage" : "single"}, {"linkage" : "ward"},
    #     {},
    # ]

    model_classes = [
        SpectralClustering, KMeans
        ]
    model_names = [
        "SpectralClustering", "KMeans"
    ]
    model_kws = [
        {}, {},
    ]

    # model_classes = [
    #     KMedoids, RobustWeightedKMeans
    #     ]
    # model_names = [
    #     "KMedoids", "RobustWeightedKMeans"
    # ]
    # model_kws = [
    #     {}, {},
    # ]

    for i_model, model_class in enumerate(model_classes):
        model_name = model_names[i_model]
        model_kw = model_kws[i_model]

        for i, (X, y) in enumerate(zip(l_data, l_labels)):

            print(" ---------------- DATASET {} ---------------- ".format(l_fname[i]))
            print(" --------------------- True k: {} --------------------- ".format(l_n_labels[i]))
            exp, fig = experiment(
                X, y,
                model_class=model_class,
                model_kw=model_kw,
            )
            exp["dataset"] = l_fname[i]
            exp["k"] = l_n_labels[i]
            exp["model"] = model_name

            # save experiment information as json
            path_saved = "./res/"
            exp_fname = "{}{}_{}_{}".format(
                path_saved, today, model_name, l_fname[i]
            )
            figtitle = "{} - {} - True k={}".format(
                l_fname[i], model_name, l_n_labels[i])
            fig.suptitle(figtitle)
            fig.savefig(exp_fname + ".png")
            json_str = json.dumps(exp, indent=2)
            with open(exp_fname + ".json", 'w', encoding='utf-8') as f:
                f.write(json_str)

    fout.close()

if __name__ == "__main__":
    main()

