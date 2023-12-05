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

def define_globals(source_number: int = 0, local=True, use_DTW=False):
    """
    Dirty way of coding in order to parallelize the processes.

    :param source_number: index in ["artificial", "real-world", "UCR"]
    :type source_number: int, optional
    :param local: to know whether the local path should be used or not,
        defaults to True
    :type local: bool, optional
    :param use_DTW: to know whether DTW should be used or not, defaults
        to False
    :type use_DTW: bool, optional
    """
    global DATA_SOURCE
    global RES_DIR, PATH, FNAME_DATASET_EXPS, FNAME_DATASET_ALL
    global SEED, K_MAX, PATH_UCR, DTW

    sources = ["artificial", "real-world", "UCR"]
    DATA_SOURCE = sources[source_number]

    RES_DIR = f'./res/{DATA_SOURCE}/'

    PATH = f"{URL_ROOT}{DATA_SOURCE}/"
    FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"
    FNAME_DATASET_ALL = f"all_datasets-{DATA_SOURCE}.txt"

    SEED = 221
    # In UCR, the max number of label in 15 so we can reduce K_MAX
    if source_number == 2:
        K_MAX = 20
    else:
        K_MAX = 25

    DTW = use_DTW and source_number == 2

    if local:
        PATH_UCR = PATH_UCR_LOCAL
    else:
        PATH_UCR = PATH_UCR_REMOTE

def metric_ts_aux(X, dist_kwargs={}, d=1, w_t=None):
    """
    Use the right metric when using UCR with DTW.

    Reshape X back to its original shape (N, T, d) because regular
    sklearn clustering methods (KMedoids, Agglomerative) need a (N, d*T)
    shape, while to compute the distances we need to go back to (N, T,
    d)

    From sklearn.metrics.pairwise_distances:

    "if metric is a callable function, it is called on each pair of
    instances (rows) and the resulting value recorded. The callable
    should take two arrays from X as input and return a value indicating
    the distance between them."

    Note that in UCR d is always 1!
    """
    dims = X.shape
    N = len(X)
    if w_t is None:
        w_t = dims[-1]
    # If using DTW and data is UCR: go from (N, T*d) to (N, T, d)
    # assuming we had either (N, T*1) or (N, T, d) to begin with
    shape = (X, w_t, d)
    X_dis = np.reshape(X, shape)

    return f_pdist(X_dis, dist_kwargs=dist_kwargs)

def experiment(
    X,
    model_class = AgglomerativeClustering,
    n_clusters_range = [i for i in range(25)],
    model_kw = {},
    scaler = StandardScaler(),
):

    exp = {}
    N = len(X)

    if N < K_MAX:
        n_clusters_range = [i for i in range(N)]

    t_start = time.time()

    # Get the dimension of X (especially important in the case of using
    # DTW in KMedoids and Agglomerative clustering)
    if DATA_SOURCE == "UCR":
        (N, w_t, d) = X.shape
    else:
        d = None
        w_t = None

    # --------------- Find the right metric for each case --------------
    def metric_ts(X, dist_kwargs={}):
        return metric_ts_aux(X, dist_kwargs=dist_kwargs, d=d, w_t=w_t)

    # Classes that do have a metric kwargs
    if model_class in [KMedoids, AgglomerativeClustering]:
        if DTW:
            model_kw["metric"] = metric_ts
        else:
            model_kw["metric"] = "euclidean"
    # Metrics that don't have a metric kwargs
    elif model_class in [KMeans, SpectralClustering]:
        # No metric kwargs in this case (and they are not used with DTW)
        pass
    elif model_class in [TimeSeriesKMeans]:
        # No metric kwargs in this case
        pass

    # --------------- Find the right shape for each case ---------------
    if model_class not in [TimeSeriesKMeans]:
        X = np.reshape(X, (N, -1))

    clusterings = generate_all_clusterings(
            X,
            model_class,
            n_clusters_range,
            DTW = model_class == TimeSeriesKMeans,
            scaler=scaler,
            model_kw=model_kw,
            time_window=None,
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
    out_fname = f'./output-run-{today}_{DATA_SOURCE}_{run_number}_{DTW}.txt'

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
        if N <= 10000 and n_labels is not None and n_labels <= 20:
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
            {"linkage" : "single"},
            {"linkage" : "average"},
        ]
    elif run_number == 1:

        model_classes = [KMedoids]
        model_names = ["KMedoids", ]
        model_kws = [{}]

    elif run_number == 2:
        if DTW:
            model_classes = [ TimeSeriesKMeans ]
            model_names = [ "TimeSeriesKMeans" ]
            model_kws = [{}]
        else:
            model_classes = [ KMeans ]
            model_names = [ "KMeans" ]
            model_kws = [{}]
    elif run_number == 4:
        model_classes = []
        model_names = []
        model_kws = []

        if not DTW:
            model_classes += [ SpectralClustering]
            model_names += ["SpectralClustering"]
            model_kws += [{}]

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
            exp['DTW'] = DTW

            # save experiment information as json
            os.makedirs(f"{RES_DIR}{model_name}_{DTW}", exist_ok=True)
            exp_fname = "{}{}_{}/{}".format(
                RES_DIR, model_name, DTW, l_fname[i]
            )
            exp["fname"] = exp_fname
            json_str = json.dumps(exp, indent=2)

            with open(exp_fname + ".json", 'w', encoding='utf-8') as f:
                f.write(json_str)

    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}")
    fout.close()

def run_process(source_number, run_number, local, use_DTW):
    define_globals(source_number, local, use_DTW)
    main(run_number)

if __name__ == "__main__":
    # source_numbers = range(3)
    source_numbers = [2]
    run_numbers = [2, 4]

    local = bool(int(sys.argv[1]))
    DTWs = [False]

    # All combination with DTW=False
    if False in DTWs:
        processes = [
            Process(target=run_process, args=(i, j, local, False))
            for i in source_numbers for j in run_numbers
        ]
    else:
        processes = []
    # Then the DTW=True is relevant only on UCR data
    if True in DTWs and 2 in source_numbers:
        processes += [
            Process(target=run_process, args=(2, j, local, True))
            for j in run_numbers
        ]

    # kick them off
    for process in processes:
        process.start()
    # now wait for them to finish
    for process in processes:
        process.join()

    # source_number = int(sys.argv[2])
    # run_number = int(sys.argv[3])

    # define_globals(source_number)
    # main(run_number)


