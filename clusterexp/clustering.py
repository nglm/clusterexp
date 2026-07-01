import os

from math import exp
from pycvi.vi import variation_information

from typing import Tuple, List

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
    Compute the Variation of Information (VI) and quality between the true clustering and each predicted clustering.

    Parameters
    ----------
    true_clusters : array-like
        The true cluster labels.
    clusterings : dict
        A dictionary of predicted clusterings, where keys are k, the number of clusters, and values are the predicted labels.

    Returns
    -------
    dict
        A dictionary of VI values for each clustering method.
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
        path_res: str,
        path_data: str
    ) -> List[Tuple[str, str]]:
    """
    From a list of experiment filenames to clustering methods and datasets.

    We assume that the experiment filenames are in the format:

    ``path/to/res/clustering_name/path/to/dataset-clustering.json`` or ``path/to/res/clustering_name/path/to/dataset-CVI.json``

    and we assume that the clustering_name doesn't contain any `/` character.

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