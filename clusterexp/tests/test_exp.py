
import numpy as np
import pytest

from clusterexp.utils import (write_json, extract_log_from_text)

from clusterexp.exp import (
    create_clusterings,
    prepare_data
)

config1 = {
    "config_data": {
        "path_data" : "example_data/",
        "path_res" : "test/test_prepare_data/",
        "max_n_samples": 1000,
        "max_n_labels": 10,
        "max_n_dims": 10,
        "exclude": ["arrhythmia", "2d-4c-no"],
        "include_only": ["artificial/"]
    },
    "config_CVI": {
        "seed": 221,
        "Hartigan": {
            "cvi": "pycvi.cvi.Hartigan"
        },
        "Inertia-sum": {
            "cvi": "pycvi.cvi.Inertia",
            "cvi_kw": {
                "reduction": "sum"
        }
        },
        "Diameter-max": {
            "cvi": "pycvi.cvi.Diameter",
            "cvi_kw": {
                "reduction": "max"
            }
        }
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
        "Agglomerative-Ward" : {
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "ward",
                "metric": "euclidean"
            },
            "fit_predict_kw": {},
            "scaler": None,
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

def test_prepare_data():

    dir = "test/test_prepare_data"
    config_fname = f"{dir}/config.json"
    write_json(config_fname, config1)

    log = prepare_data(config_fname)

    assert isinstance(log, dict)
    assert "config_data" in log
    assert "log_data" in log
    assert log["config_data"] == config1["config_data"]

    # Make sure that the text log file contain 2 logs and that the last log corresponds to the log dictionary
    l_log_extracted = extract_log_from_text(f"{log['log_data']['log_fname']}.txt")

    assert isinstance(l_log_extracted, list)
    assert len(l_log_extracted) == 2
    assert l_log_extracted[-1] == log

def test_create_clusterings():

    dir = "test/test_create_clusterings"
    config_fname = f"{dir}/config.json"
    config2 = config1.copy()
    config2["config_data"]["path_res"] = f"{dir}/"
    write_json(config_fname, config2)

    log = create_clusterings(config_fname)

    assert isinstance(log, dict)
    assert "config_data" in log
    assert "log_data" in log
    assert "log_clustering" in log
    assert "config_clustering" in log
    assert log["config_data"] == config2["config_data"]

    # Make sure that the text log file contain 2 logs and that the last log corresponds to the log dictionary
    l_log_extracted = extract_log_from_text(f"{log['log_clustering']['log_fname']}.txt")

    assert isinstance(l_log_extracted, list)
    assert len(l_log_extracted) == 2
    assert l_log_extracted[-1] == log