#!/usr/bin/env python3

from multiprocessing import Process
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from tslearn.clustering import TimeSeriesKMeans
from datetime import date
import sys, os
import time
import json
from mpl_toolkits.mplot3d import Axes3D

from pycvi.cluster import generate_all_clusterings
from pycvi.compute_scores import f_pdist

from clusterexp.utils import (
    get_data_labels, get_data_labels_UCR, write_list_datasets, get_fname,
    UNKNOWN_K, INVALID, URL_ROOT, PATH_UCR_LOCAL, PATH_UCR_REMOTE
)

import warnings
warnings.filterwarnings("ignore")

def define_globals(source_number: int = 0, local=True):
    global DATA_SOURCE
    global RES_DIR, PATH, FNAME_DATASET_EXPS, FNAME_DATASET_ALL
    global SEED, METRIC, K_MAX, PATH_UCR

    sources = ["artificial", "real-world", "UCR"]
    DATA_SOURCE = sources[source_number]

    RES_DIR = f'./res/{DATA_SOURCE}/'

    PATH = f"{URL_ROOT}{DATA_SOURCE}/"
    FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"
    FNAME_DATASET_ALL = f"all_datasets-{DATA_SOURCE}.txt"

    SEED = 221
    K_MAX = 25

    def metric_dtw(X, dist_kwargs={}, DTW: bool = True):
        dims = X.shape
        if len(dims) == 2 and DTW:
            X = np.expand_dims(X, axis=2)
        return f_pdist(X, dist_kwargs=dist_kwargs)

    if DATA_SOURCE == "UCR":
        METRIC = metric_dtw
    else:
        METRIC = "euclidean"

    if local:
        PATH_UCR = PATH_UCR_LOCAL
    else:
        PATH_UCR = PATH_UCR_REMOTE

def experiment(
    X,
    model_class = AgglomerativeClustering,
    n_clusters_range = [i for i in range(K_MAX)],
    model_kw = {},
    scaler = StandardScaler(),
):

    exp = {}

    N = len(X)
    if N < K_MAX:
        n_clusters_range = [i for i in range(K_MAX)]

    t_start = time.time()

    DTW = DATA_SOURCE == "UCR"
    if DTW and model_class != TimeSeriesKMeans:
        X = np.squeeze(X)

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

def main(run_number: int = 0):
    """
    Compute and store clusterings for all datasets for a given set of
    clustering methods.
    """

    np.random.seed(SEED)

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = f'./output-run-{today}_{DATA_SOURCE}_{run_number}.txt'

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # --------- Get datasets and labels -----------
    with open(RES_DIR + FNAME_DATASET_ALL) as f:
        all_datasets = f.read().splitlines()

    l_data = []
    l_labels = []
    l_n_labels = []
    l_fname = []
    # Keep only datasets with known K and no missing values
    l_fname_all = [
        f for f in all_datasets
        if f not in UNKNOWN_K + INVALID
    ]
    for fname in l_fname_all:
        #print(fname, flush=True)
        if DATA_SOURCE == "UCR":
            f = f"{PATH_UCR}{get_fname(fname, data_source=DATA_SOURCE)}"
            data, labels, n_labels, _ = get_data_labels_UCR(f, path=PATH)
        else:
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
    # Write dataset list that is kept in the end
    write_list_datasets(RES_DIR + FNAME_DATASET_EXPS, l_fname)

    # ------ Run experiments for all datasets and all scores -----------
    scaler = StandardScaler()
    scaler_name = "StandardScaler"

    if run_number == 0:

        model_classes = [
            AgglomerativeClustering, AgglomerativeClustering,
            ]
        model_names = [
            "AgglomerativeClustering-Single", "AgglomerativeClustering-Average",
        ]
        model_kws = [
            {"linkage" : "single", "metric" : METRIC},
            {"linkage" : "average", "metric" : METRIC},
        ]
    elif run_number == 1:

        model_classes = [ SpectralClustering, KMedoids ]
        model_names = [ "SpectralClustering", "KMedoids", ]
        model_kws = [ {}, {"metric" : METRIC}]

    elif run_number == 2:
        if DATA_SOURCE == "UCR":
            model_classes = [ TimeSeriesKMeans ]
            model_names = [ "TimeSeriesKMeans" ]
            model_kws = [ {}]
        else:
            model_classes = [ KMeans ]
            model_names = [ "KMeans" ]
            model_kws = [ {}]

    # model_classes = [
    #     KMedoids,
    #     ]
    # model_names = [
    #     "KMedoids",
    # ]
    # model_kws = [
    #     {"metric" : METRIC},
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
                n_clusters_range=[i for i in range(K_MAX)],
                model_class=model_class,
                model_kw=model_kw,
                scaler=scaler,
            )
            exp["dataset"] = l_fname[i]
            exp["k"] = l_n_labels[i]
            exp["scaler"] = scaler_name
            exp["model"] = model_name
            exp["model_kw"] = {k : str(v) for k, v in model_kw.items()}
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

def run_process(source_number, run_number):
    define_globals(source_number)
    main(run_number)

if __name__ == "__main__":
    source_numbers = range(3)
    run_numbers = range(3)
    processes = [
        Process(target=run_process, args=(i, j))
        for i in source_numbers for j in run_numbers
    ]

    # kick them off
    for process in processes:
        process.start()
    # now wait for them to finish
    for process in processes:
        process.join()
    # source_number = int(sys.argv[1])
    # run_number = int(sys.argv[1])

    # define_globals(source_number)
    # main(run_number)


