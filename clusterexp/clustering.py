"""Helpers for clustering experiments."""

import os

from math import exp
from pycvi.vi import variation_information

from typing import Tuple, List

from .utils import interpret_dict

def f_quality(VI: float) -> float:
    """
    Quality of a clustering.

    The quality of a predicted clustering is based on the VI between the
    true and predicted clusterings. We define the quality as:

    :math:`quality = exp(-2*VI)`

    Which means that the quality is between 0 and 1, with higher quality
    values meaning better clusterings.

    Parameters
    ----------
    VI : float
        The Variation of Information between the true clustering and the
        predicted clustering

    Returns
    -------
    float
        The quality of the predicted clustering.
    """
    return exp(-2*VI)


def compute_VI_quality(true_clusters, clusterings: dict) -> Tuple[dict, dict]:
    """
    Compute VI and quality for each predicted clustering.

    Parameters
    ----------
    true_clusters : array-like
        The true cluster labels.
    clusterings : dict
        A dictionary of predicted clusterings, where keys are k, the number of clusters, and values are the predicted labels.

    Returns
    -------
    Tuple[dict, dict]
        Two dictionaries keyed like ``clusterings``: the first contains
        VI values, and the second contains transformed quality scores.
    """
    # ------------------------ Compute VIs ------------------------------
    # Compute VI between the true clustering and each clustering
    # obtained with the different clustering methods with the
    # real number of clusters
    VIs = {}
    qualities = {}

    for k, clustering in clusterings.items():
        if clustering is None:
            VIs[k] = None
        else:
            VIs[k] = variation_information(true_clusters, clustering)
            qualities[k] = f_quality(VIs[k])
    return VIs, qualities

def decompose_exp_fnames(
        exp_fnames: List[str],
    ) -> List[Tuple[str, str]]:
    """
    From a list of experiment filenames to clustering methods and datasets.

    We assume that the experiment filenames are in the format:

    ``path/to/res/clustering_name/path/to/dataset-clustering.json`` or ``path/to/res/clustering_name/path/to/dataset-CVI.json``

    and we assume that the clustering_name doesn't contain any `/` character.

    Note that path_data doesn't appear in the experiment filenames, only the path to the dataset relative to path_data.

    Parameters
    ----------
    exp_fnames : list
        A list of experiment filenames.

    Returns
    -------
    List[Tuple[str, str]]
        A list of tuples containing the clustering methods and datasets.
    """

    path_res = os.path.commonpath(exp_fnames)

    result = []

    for fname in exp_fnames:
        # 1: remove path_res with split(path_res)[-1]
        # 2. and the "/" between path_res and the rest of the path [1:]
        no_path_res = fname.split(path_res)[-1][1:]

        clustering_method, path_fname = no_path_res.split("/", 1)
        if path_fname.endswith("-clustering.json"):
            path_dataset = path_fname[:-len("-clustering.json")]
        elif path_fname.endswith("-CVI.json"):
            path_dataset = path_fname[:-len("-CVI.json")]
        result.append((clustering_method, path_dataset))
    return result

def group_exp_by_dataset(exp_fnames: List[str]) -> dict:
    """
    Group experiments by dataset.

    Note that the root of the datasets (path_data) is not included here

    The dictionary is sorted by dataset name (yes, I am aware that
    dictionaries are not really ordered, it's just a plus) and the list
    of experiments for each dataset is sorted by experiment filename.

    Parameters
    ----------
    exp_fnames : list
        A list of experiment filenames.

    Returns
    -------
    dict
        A dictionary where keys are datasets and values are lists of experiment filenames.
    """
    # decompose experiments (clustering_method, path_dataset)
    decomposed_exp = decompose_exp_fnames(exp_fnames)

    # Find all datasets
    datasets = set([path_dataset for _, path_dataset in decomposed_exp])
    datasets = sorted(list(datasets))

    # Group experiments by dataset
    grouped_exp = {}
    for d in datasets:
        grouped_exp[d] = sorted([
            fname for fname, decomp_exp in zip(exp_fnames, decomposed_exp)
            if decomp_exp[1] == d
        ])

    return grouped_exp

def filter_experiments(
        exp_fnames: List[str],
        **constraints,
    ) -> Tuple[dict, dict]:
    """
    Filter experiments (and datasets) based on clustering quality.

    Note that if both "best_q_true" and "best_q_best" are set to True, then only the experiments that are the best for both true and best qualities will be kept.

    If you want to be able to filter both based on the best q_best and q_true
    then you should call this function twice, once with best_q_true=True and once with best_q_best=True, and then take the intersection of the two sets of kept experiments.

    Possible constraints include:
    - `best_q_true_only` : bool, optional: Keep only the best clustering method per dataset based on the true quality (default: False)
    - `best_q_best_only` : bool, optional: Keep only the best clustering method per dataset based on the best quality (default: False)
    - `quality_true_min` : float, optional: Keep only experiments with a quality_true above a given threshold if given (default: 0)
    - `quality_best_min` : float, optional: Keep only experiments with a quality_best above a given threshold if given (default: 0)

    Parameters
    ----------
    exp_fnames : list
        A list of experiment filenames.
    **constraints : dict
        Constraints to filter experiments.

    Returns
    -------
    Tuple[dict, dict]
        A tuple containing two dictionaries:
        - The first dictionary contains the kept and dropped experiments with reasons for dropping.
            - `kept_experiments`: a list of kept experiment filenames
            - `dropped_experiments`: a dictionary containing the dropped experiments with reasons for dropping
                - `not_best_q_true`: a list of dropped experiment filenames whose true quality is not the best for the dataset
                - `not_best_q_best`: a list of dropped experiment filenames whose best quality is not the best for the dataset
                - `quality_true_min`: a list of dropped experiment filenames whose true quality is below the threshold
                - `quality_best_min`: a list of dropped experiment filenames whose best quality is below the threshold
        - The second dictionary contains the kept and dropped datasets.
            - `kept_datasets`: a list of kept dataset names
            - `dropped_datasets`: a list of dropped dataset names

    """

    kept_datasets = []
    dropped_datasets = []

    kept_exp = []
    best_q_true = []
    best_q_best = []
    dropped_exp = {
        "not_best_q_true": [],
        "not_best_q_best": [],
        "quality_min": [],
        "quality_true_min" : [],
        "quality_best_min" : [],
    }

    # Group experiments by dataset {dataset: [exp_fnames]}
    grouped_exp = group_exp_by_dataset(exp_fnames)

    # For each dataset, go through all its experiments and filter
    for dataset, exps in grouped_exp.items():

        # Qualities for true and best clusterings for this dataset
        qualities_true = []
        qualities_best = []
        keeps = []

        # Go through all experiments for this dataset, find their VIs
        for exp in exps:

            # a priori keep this experiment until we find a reason to drop it
            keep = True

            # Load the experiment
            exp_log = interpret_dict(exp)
            k_true = exp_log["k_true"]

            # Find true and best qualities for this experiment
            quality_true = exp_log["qualities"][k_true]
            quality_best = max(exp_log["qualities"].values())
            qualities_true.append(quality_true)
            qualities_best.append(quality_best)

            # Drop based on quality thresholds if constraint is given
            if quality_true < constraints.get("quality_true_min", 0):
                dropped_exp["quality_true_min"].append(exp)
                keep = False
            if quality_best < constraints.get("quality_best_min", 0):
                dropped_exp["quality_best_min"].append(exp)
                keep = False

            keeps.append(keep)


        # Find the best clustering for this dataset
        best_q_true_idx = max(enumerate(qualities_true), key=lambda x: x[1])[0]
        best_q_true.append(exps[best_q_true_idx])

        # Drop based on best true quality if constraint is given
        if constraints.get("best_q_true_only", False):
            for i, exp in enumerate(exps):
                if i != best_q_true_idx:
                    dropped_exp["not_best_q_true"].append(exp)
                    keeps[i] = False

        # Find the best clustering for this dataset
        best_q_best_idx = max(enumerate(qualities_best), key=lambda x: x[1])[0]
        best_q_best.append(exps[best_q_best_idx])

        # Drop based on best best quality if constraint is given
        if constraints.get("best_q_best_only", False):
            for i, exp in enumerate(exps):
                if i != best_q_best_idx:
                    dropped_exp["not_best_q_best"].append(exp)
                    keeps[i] = False

        # Now add kept experiments for this dataset to the kept_exp list
        for i, exp in enumerate(exps):
            if keeps[i]:
                kept_exp.append(exp)

        # Keep this dataset if at least one experiment was kept for this dataset
        if any(keeps):
            kept_datasets.append(dataset)
        else:
            dropped_datasets.append(dataset)

    filtered_exp = {
        "best_q_true" : best_q_true,
        "best_q_best" : best_q_best,
        "kept_experiments": kept_exp,
        "dropped_experiments": dropped_exp,
    }

    filtered_datasets = {
        "kept_datasets": kept_datasets,
        "dropped_datasets": dropped_datasets,
    }
    return filtered_exp, filtered_datasets
