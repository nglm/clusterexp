#!/usr/bin/env python3

import os
import shutil
import sys
import numpy as np
import json
from datetime import date
from pycvi.vi import variational_information

from clusterexp.utils import load_data_from_github, get_data_labels


RES_DIR = './res/'
URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'
PATH = URL_ROOT + "artificial/"
FNAME_DATASET_EXPS = "datasets_experiments.txt"
FNAME_DATASET_FILTERED = "datasets_filtered.txt"
SELECTED_DIR = RES_DIR + "Selected/"
VI_MAX = 0.2

def main():
    """
    Filter dataset for which we managed to have a good enough clustering
    based of the variation of information with the true clustering
    """

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = './output-filter-' + today + ".txt"
    selected_dataset = []

    fout = open(out_fname, 'wt')
    sys.stdout = fout

    # Find (non-filtered) selected experimentsd
    fnames = [
        SELECTED_DIR + f[:-5]
        for f in os.listdir(SELECTED_DIR) if ".json" in f
        ]

    # For each selected experiments, filter out those who don't have
    # a good enough VI
    for fname in fnames:

        with open(fname + ".json") as f_json:
            exp = json.load(f_json)

        vi = exp["VI"]

        if vi < VI_MAX:
            selected_dataset.append(exp["dataset"] + "\n")

        print(f'{fname}  |   {vi}', flush=True)

    with open(FNAME_DATASET_FILTERED, "w") as f:
        f.writelines(selected_dataset)

    fout.close()

if __name__ == "__main__":
    main()