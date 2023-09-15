#!/usr/bin/env python3

import os
import shutil
import numpy as np
import json
from pycvi.vi import variational_information

from clusterexp.utils import load_data_from_github, get_data_labels


RES_DIR = './res/'
URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"
FNAME_DATASET_EXPS = "datasets_experiments.txt"
SELECTED_DIR = RES_DIR + "Selected/"

def main():
    """
    Select the most relevant clustering algorithm for each dataset
    based of the variation of information with the true clustering
    """

    # List of directories, corresponding to clustering methods
    list_dirs = [dname.strip() for dname in next(os.walk(RES_DIR))[1] if dname != "Selected"]
    with open(RES_DIR + FNAME_DATASET_EXPS) as f:
        datasets = f.read().splitlines()

    for d in datasets:

        # For each dataset, find all experiments working on this dataset
        # but using different clustering methods (by filtering on the
        # filename)
        fnames = []
        for dir in list_dirs:

            dir_fnames = [f.name for f in os.scandir(RES_DIR + dir)]
            # get full directory + filename without the extension
            fnames += [
                dir + "/" + fname[:-5] for fname in dir_fnames
                if d + ".json" in fname
            ]

        # Storing the variation of information of each experiment
        VIs = []

        # --------- Get datasets and labels -----------
        data, labels, n_labels, meta = get_data_labels(d, path=PATH)
        classes = np.unique(labels)

        # true clusters
        true_clusters = [labels == classes[i] for i in range(n_labels)]

        for fname in fnames:

            with open(RES_DIR + fname + ".json") as f_json:
                exp = json.load(f_json)

            # all clusterings of this experiment
            clusterings = exp['clusterings']

            # best clusters
            best_clusters = clusterings[n_labels]

            # Compute VI between the true clustering and each clustering
            # obtained with the different clustering method with the
            # real number of clusters
            VIs.append(variational_information(true_clusters, best_clusters))

        # Keep only the clustering method with the lowest VI and copy
        # the experiment file and the figures to another folder
        argbest = np.argmin(VIs)
        fname = fnames[argbest]
        shutil.copy(fname + ".json", SELECTED_DIR)
        shutil.copy(fname + ".png", SELECTED_DIR)

if __name__ == "__main__":
    main()