#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from datetime import date
import sys, os
import time
import json
from mpl_toolkits.mplot3d import Axes3D

from pycvi.cluster import generate_all_clusterings

from clusterexp.utils import get_data_labels, UNKNOWN_K, URL_ROOT

import warnings
warnings.filterwarnings("ignore")

DATA_SOURCE = "artificial"
DATA_SOURCE = "real-world"

RES_DIR = f'./res/{DATA_SOURCE}/'

PATH = f"{URL_ROOT}{DATA_SOURCE}/"
FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"

SEED = 221

def experiment(
    X,
    model_class = AgglomerativeClustering,
    n_clusters_range = [i for i in range(25)],
    model_kw = {},
    scaler = StandardScaler(),
):

    exp = {}

    N = len(X)
    if N > 10000:
        print("Dataset too big {}".format(N))
    else:

        t_start = time.time()

        clusterings = generate_all_clusterings(
                X,
                model_class,
                n_clusters_range,
                DTW=False,
                scaler=scaler,
                model_kw=model_kw,
            )

        t_end = time.time()
        dt = t_end - t_start

        # store clustering information
        exp["clusterings"] = clusterings[0]
        exp["time"] : dt

        print(f"Clusterings generated in: {dt:.2f}s")

    return exp

def main():
    """
    Compute and store clusterings for all datasets for a given set of
    clustering methods.
    """

    np.random.seed(SEED)

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = './output-run-' + today + ".txt"

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # --------- Get datasets and labels -----------
    with open(RES_DIR + FNAME_DATASET_EXPS) as f:
        all_datasets = f.read().splitlines()

    l_data = []
    l_labels = []
    l_n_labels = []
    l_fname = []
    l_fname_all = [f for f in all_datasets if f not in UNKNOWN_K]
    for fname in l_fname_all:
        #print(fname, flush=True)
        data, labels, n_labels, meta = get_data_labels(fname, path=PATH)
        N = len(data)
        if N <= 10000 and n_labels <= 20:
            l_data.append(data)
            l_labels.append(labels)
            l_n_labels.append(n_labels)
            l_fname.append(fname)
            print(
                "Dataset: {}  | Shape: {}  | #labels: {}".format(
                    fname, data.shape, n_labels
                ), flush=True
            )

    # ------ Run experiments for all datasets and all scores -----------
    scaler = StandardScaler()
    scaler_name = "StandardScaler"
    # model_classes = [
    #     AgglomerativeClustering, AgglomerativeClustering,
    #     ]
    # model_names = [
    #     "AgglomerativeClustering-Single", "AgglomerativeClustering-Ward",
    # ]
    # model_kws = [
    #     {"linkage" : "single"}, {"linkage" : "ward"},
    # ]

    model_classes = [ SpectralClustering ]
    model_names = [ "SpectralClustering" ]
    model_kws = [ {}]

    # model_classes = [ KMeans ]
    # model_names = [ "KMeans" ]
    # model_kws = [ {}]

    # model_classes = [
    #     KMedoids,
    #     ]
    # model_names = [
    #     "KMedoids",
    # ]
    # model_kws = [
    #     {},
    # ]

    t_start = time.time()
    for i_model, model_class in enumerate(model_classes):
        model_name = model_names[i_model]
        model_kw = model_kws[i_model]

        print(f" ======= {model_name} =======", flush=True)

        for i, (X, y) in enumerate(zip(l_data, l_labels)):

            print(f"\n{l_fname[i]}", flush=True)
            exp = experiment(
                X,
                n_clusters_range=[i for i in range(25)],
                model_class=model_class,
                model_kw=model_kw,
                scaler=scaler,
            )
            exp["dataset"] = l_fname[i]
            exp["k"] = l_n_labels[i]
            exp["scaler"] = scaler_name
            exp["model"] = model_name
            exp["model_kw"] = model_kw
            exp['seed'] = SEED

            # save experiment information as json
            os.makedirs(f"{RES_DIR}{model_name}", exist_ok=True)
            exp_fname = "{}{}/{}".format(
                RES_DIR, model_name, l_fname[i]
            )
            exp["fname"] = exp_fname
            json_str = json.dumps(exp, indent=2)

            with open(exp_fname + ".json", 'w', encoding='utf-8') as f:
                f.write(json_str)

    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}")
    fout.close()

if __name__ == "__main__":
    main()


