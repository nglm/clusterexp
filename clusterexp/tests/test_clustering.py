
import numpy as np
import pytest


from pycvi.cluster import get_clustering

from clusterexp.config import interpret_config
from clusterexp.clustering import (
    decompose_exp_fnames, f_quality, compute_VI_quality, group_exp_by_dataset
)
from clusterexp.exp import create_clusterings

def test_f_quality():
    # Test that f_quality returns the expected values for known VI values
    quality = f_quality(0.0)
    assert np.isclose(quality, 1.0)
    assert isinstance(quality, float)

def test_compute_VI_quality():
    # Test that compute_VI_quality returns the expected values for known true and predicted clusterings
    true_clusters = get_clustering(np.array([0, 0, 1, 1]))
    clusterings = {
        2: get_clustering(np.array([0, 0, 1, 1])),
        3: get_clustering(np.array([0, 0, 1, 2])),
        4: get_clustering(np.array([0, 1, 2, 3]))
    }
    VIs, qualities = compute_VI_quality(true_clusters, clusterings)
    assert isinstance(VIs, dict)
    assert isinstance(qualities, dict)
    assert all(isinstance(v, float) for v in VIs.values())
    assert all(isinstance(q, float) for q in qualities.values())

def test_decompose_exp_fnames():
    exp_fnames = [
        "res/KMeans/artificial/2d-4c-no-clustering.json",
        "res/Agglomerative-Single/artificial/2d-4c-no-clustering.json",
        "res/Agglomerative-Ward/artificial/2d-4c-no-clustering.json",
        "res/KMeans/artificial/2d-4c-no-CVI.json",
        "res/Agglomerative-Single/artificial/2d-4c-no-CVI.json",
        "res/Agglomerative-Ward/artificial/2d-4c-no-CVI.json"
    ]

    expected_output = [
        ("KMeans", "artificial/2d-4c-no"),
        ("Agglomerative-Single", "artificial/2d-4c-no"),
        ("Agglomerative-Ward", "artificial/2d-4c-no"),
        ("KMeans", "artificial/2d-4c-no"),
        ("Agglomerative-Single", "artificial/2d-4c-no"),
        ("Agglomerative-Ward", "artificial/2d-4c-no")
    ]
    output = decompose_exp_fnames(exp_fnames)
    assert output == expected_output

def test_group_exp_by_dataset():

    config1 = {
    "config_data": {
        "path_data" : "example_data/",
        "path_res" : "test/test_prepare_data/",
    },
    "config_clustering" : {
        "seed" : 221,
        "k_range" : [1, 25],
        "KMeans" : {
            "model" : "sklearn.cluster.KMeans",
            "model_kw" : {},
            "fit_predict_kw" : {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "Agglomerative-Single" : {
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "single",
                "metric": "euclidean"
            },
            "fit_predict_kw": {},
            "scaler_kw": {}
        },
        "KMedoids" : {
            "model": "kmedoids.KMedoids",
            "model_kw": {
                "metric": "euclidean",
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
    }
    }

    config1 = interpret_config(config1)
    log = create_clusterings(config1)

    exp = log["log_clustering"]["path_exp"]
    datasets = log["log_data"]["kept_datasets"]

    grouped_exp = group_exp_by_dataset(exp)

    # Is dict
    assert isinstance(grouped_exp, dict)
    # Each dataset is represented
    assert all(d in grouped_exp for d in datasets)
    assert len(grouped_exp) == len(datasets)
    # Each experiment is represented
    assert all(any(exp in grouped_exp[d] for d in datasets) for exp in exp)
