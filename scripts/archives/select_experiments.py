#!/usr/bin/env python3

import os
import shutil
import sys
import numpy as np
import json
from datetime import date
from pycvi.vi import variational_information

from clusterexp.utils import (
    load_data_from_github, get_data_labels, get_list_datasets,
    get_list_exp, URL_ROOT
)


DATA_SOURCE = "artificial"
#DATA_SOURCE = "real-world"

RES_DIR = f'./res/{DATA_SOURCE}/'
PATH = f"{URL_ROOT}{DATA_SOURCE}/"
FNAME_DATASET_EXPS = f"datasets_experiments-{DATA_SOURCE}.txt"

SELECTED_DIR = RES_DIR + "Selected/"

def main():
    """
    Select the most relevant clustering algorithm for each dataset
    based of the variation of information with the true clustering
    """

    datasets = get_list_datasets(RES_DIR + FNAME_DATASET_EXPS)

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = './output-select-' + today + ".txt"

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    for d in datasets:

        # For each dataset, find all experiments working on this dataset
        # but using different clustering methods (by filtering on the
        # filename)
        fnames = get_list_exp(dataset_name=d, res_dir=RES_DIR)

        # Storing the variation of information of each experiment
        VIs = []

        # --------- Get datasets and labels -----------
        data, labels, n_labels, meta = get_data_labels(d, path=PATH)
        classes = np.unique(labels)

        # true clusters
        # List[List[int]]
        indices = np.arange(len(data))
        true_clusters = [
            indices[labels == classes[i]] for i in range(n_labels)
        ]

        print(fnames)

        for fname in fnames:

            with open(RES_DIR + fname + ".json") as f_json:
                exp = json.load(f_json)

            # all clusterings of this experiment
            clusterings = exp['clusterings']

            # best clusters (keys become string when saved/loaded)
            best_clusters = clusterings[str(n_labels)]

            # Compute VI between the true clustering and each clustering
            # obtained with the different clustering method with the
            # real number of clusters
            VI = variational_information(true_clusters, best_clusters)
            VIs.append(VI)
            print(fname, VI)

            # Add VI to the exp file
            exp["VI"] = VI
            json_str = json.dumps(exp, indent=2)
            with open(RES_DIR+fname + ".json", 'w', encoding='utf-8') as f_json:
                f_json.write(json_str)

        # Keep only the clustering method with the lowest VI and copy
        # the experiment file and the figures to another folder
        argbest = np.argmin(VIs)
        fname = fnames[argbest]
        print(fname)
        shutil.copy(RES_DIR + fname + ".json", SELECTED_DIR)
        shutil.copy(RES_DIR + fname + ".png", SELECTED_DIR)

    fout.close()

if __name__ == "__main__":
    main()