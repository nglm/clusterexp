
import numpy as np
import pytest


from pycvi.cluster import get_clustering

from clusterexp.config import interpret_config
from clusterexp.clustering import (
    decompose_exp_fnames, f_quality, compute_VI_quality, group_exp_by_dataset, filter_experiments
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


def test_filter_experiments():

    config1 = {
    "config_data": {
        "path_data" : "example_data/",
        "path_res" : "test/test_prepare_data/",
        "max_n_labels" : 25,
    },
    "config_clustering" : {
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

    # -------------- With no constraints ------------------
    filtered_exp, filtered_datasets = filter_experiments(exp)

    # Check types
    assert isinstance(filtered_exp, dict)
    assert isinstance(filtered_datasets, dict)
    assert isinstance(filtered_exp["kept_experiments"], list)
    assert isinstance(filtered_exp["dropped_experiments"], dict)
    assert all(isinstance(v, list) for v in filtered_datasets.values())

    # Check that best q_true and q_max are present
    assert len(filtered_exp["best_q_true"]) == len(datasets)
    assert len(filtered_exp["best_q_max"]) == len(datasets)

    # Check that everything is kept when no constraints are given
    assert len(filtered_exp["kept_experiments"]) == len(exp)
    assert set(filtered_exp["kept_experiments"]) == set(exp)
    assert len(filtered_datasets["kept_datasets"]) == len(datasets)
    assert set(filtered_datasets["kept_datasets"]) == set(datasets)

    # Check that dropped lists are empty when no constraints are given
    assert all(v == [] for v in filtered_exp["dropped_experiments"].values())
    assert len(filtered_datasets["dropped_datasets"]) == 0

    # -------------- With constraints ---------------------
    filtered_exp, filtered_datasets = filter_experiments(
        exp, best_q_true_only=True
    )

    # Check types
    assert isinstance(filtered_exp, dict)
    assert isinstance(filtered_datasets, dict)
    assert isinstance(filtered_exp["kept_experiments"], list)
    assert isinstance(filtered_exp["dropped_experiments"], dict)
    assert all(isinstance(v, list) for v in filtered_datasets.values())

    # Check that best q_true and q_max are present
    assert len(filtered_exp["best_q_true"]) == len(datasets)
    assert len(filtered_exp["best_q_max"]) == len(datasets)

    # Check that all datasets are kept
    assert len(filtered_datasets["dropped_datasets"]) == 0
    assert len(filtered_datasets["kept_datasets"]) == len(datasets)
    assert set(filtered_datasets["kept_datasets"]) == set(datasets)

    # Assert that there are as many kept experiments as there are datasets
    # since we are keeping only the best experiment for each dataset
    assert len(filtered_exp["kept_experiments"]) == len(datasets)