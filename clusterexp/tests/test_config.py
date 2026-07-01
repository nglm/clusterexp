import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy
import os
import pytest

import pycvi
from pycvi.cvi import Hartigan

from ..config import (
    CONFIG_DATA_BASE, CONFIG_CLUSTERING_BASE,
    CONFIG_CLUSTERING_TIME_SERIES_BASE,
    CONFIG_CVI_BASE, CONFIG_DEFAULT_VALUES, get_models_config,
    make_default_config, add_default, check_config,
    interpret_saved_config, load_config_as_dict,
    get_mandatory_keys,
)
from ..utils import write_json, load_json

config_1 = {
    "VI_max": 0.2,
    "seed": 221,
    "k_range": [1, 25],
    "KMeans": {
        "model_class": sklearn.cluster.KMeans,
        "model_kw": {"random_state" : numpy.random.RandomState(211)},
        "fit_predict_kw": {},
        "scaler": sklearn.preprocessing.StandardScaler,
        "scaler_kw": {}
    },
    "some_array" : numpy.ones((3,2)),
    "some_function" : numpy.ones,
    "some_class" : sklearn.cluster.KMeans
}

config_cvi = {
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
  }
}

config_clustering = {
  "config_clustering": {
    "VI_max": 0.2,
    "seed": 221,
    "k_range": [
      1,
      25
    ],
    "KASBA": {
      "model": "aeon.clustering.KASBA",
      "model_kw": {},
      "fit_predict_kw": {},
      "scaler": "sklearn.preprocessing.StandardScaler",
      "scaler_kw": {}
    },
    "TimeSeriesKMedoids": {
      "model": "aeon.clustering.TimeSeriesKMedoids",
      "model_kw": {},
      "fit_predict_kw": {},
      "scaler": "sklearn.preprocessing.StandardScaler",
      "scaler_kw": {}
    },
    "Agglomerative-Single": {
      "model": "sklearn.cluster.AgglomerativeClustering",
      "model_kw": {
        "linkage": "single",
        "metric": "pycvi.dist.time_series_metric_with_sklearn"
      },
      "fit_predict_kw": {},
      "scaler": "sklearn.preprocessing.StandardScaler",
      "scaler_kw": {}
    }
  }
}

def test_constants():
    assert isinstance(CONFIG_DATA_BASE, dict)
    assert isinstance(CONFIG_CLUSTERING_BASE, dict)
    assert isinstance(CONFIG_CLUSTERING_TIME_SERIES_BASE, dict)
    assert isinstance(CONFIG_CVI_BASE, dict)
    assert isinstance(CONFIG_DEFAULT_VALUES, dict)

def test_get_mandatory_keys():
    mandatory_keys = get_mandatory_keys()
    assert isinstance(mandatory_keys, dict)
    assert "config_data" in mandatory_keys
    assert "config_clustering" in mandatory_keys
    assert "config_CVI" in mandatory_keys

def test_make_default_config():

    dir = "test/test_make_default_config"
    make_default_config(f"{dir}")
    assert os.path.isfile(f"{dir}/config-data.json")
    assert os.path.isfile(f"{dir}/config-clustering.json")
    assert os.path.isfile(f"{dir}/config-clustering-time_series.json")
    assert os.path.isfile(f"{dir}/config-CVI.json")

def test_add_default():
    dir = "test/test_add_default"
    make_default_config(f"{dir}")

    config_data = load_json(f"{dir}/config-data.json")
    config_clustering = load_json(f"{dir}/config-clustering.json")
    config_clustering_time_series = load_json(f"{dir}/config-clustering-time_series.json")
    config_CVI = load_json(f"{dir}/config-CVI.json")

    # The data and clustering configs should be unchanged after adding default values
    # because they already had values for every key
    config_data_res = add_default(config_data)
    assert config_data == config_data_res
    assert check_config(config_data_res)

    config_clustering_res = add_default(config_clustering)
    assert config_clustering == config_clustering_res
    assert check_config(config_clustering_res)

    config_clustering_time_series_res = add_default(config_clustering_time_series)
    assert config_clustering_time_series == config_clustering_time_series_res
    assert check_config(config_clustering_time_series_res)

    # Config CVI is different
    config_cvi_res = add_default(config_CVI)
    assert config_CVI != config_cvi_res
    assert check_config(config_cvi_res)

    assert os.path.isfile(f"{dir}/config-data.json")
    assert os.path.isfile(f"{dir}/config-clustering.json")
    assert os.path.isfile(f"{dir}/config-clustering-time_series.json")
    assert os.path.isfile(f"{dir}/config-CVI.json")

def test_interpret_saved_config():
    dir = "test/test_interpret_saved_config"
    make_default_config(f"{dir}")
    config_clustering = load_json(f"{dir}/config-clustering.json")
    config_CVI = load_json(f"{dir}/config-CVI.json")

    config_CVI_interpreted = interpret_saved_config(config_CVI)
    config_clustering_interpreted = interpret_saved_config(config_clustering)

    assert check_config(config_clustering_interpreted)
    assert check_config(config_CVI_interpreted)
    config_clustering_interpreted
    assert config_CVI_interpreted["config_CVI"]["Hartigan"]["cvi"] == pycvi.cvi.Hartigan
    assert config_CVI_interpreted["config_CVI"]["Hartigan"]["cvi"] == Hartigan
    assert config_clustering_interpreted["config_clustering"]["KMeans"]["model"] == sklearn.cluster.KMeans
    assert config_clustering_interpreted["config_clustering"]["KMeans"]["model"] == KMeans


def test_load_config_as_dict():
    dir = "test/test_load_config_as_dict"
    make_default_config(f"{dir}")

    config_CVI_loaded = load_config_as_dict(f"{dir}/config-CVI.json")
    config_clustering_loaded = load_config_as_dict(f"{dir}/config-clustering.json")

    assert check_config(config_clustering_loaded)
    assert check_config(config_CVI_loaded)
    assert config_CVI_loaded["config_CVI"]["Hartigan"]["cvi"] == pycvi.cvi.Hartigan
    assert config_CVI_loaded["config_CVI"]["Hartigan"]["cvi"] == Hartigan
    assert config_clustering_loaded["config_clustering"]["KMeans"]["model"] == sklearn.cluster.KMeans
    assert config_clustering_loaded["config_clustering"]["KMeans"]["model"] == KMeans

def test_get_models_config():
    models_clustering = get_models_config(config_clustering)

    expected_models_clustering = {
        "config_clustering": {
            "KASBA": {
            "model": "aeon.clustering.KASBA",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
            },
            "TimeSeriesKMedoids": {
            "model": "aeon.clustering.TimeSeriesKMedoids",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
            },
            "Agglomerative-Single": {
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "single",
                "metric": "pycvi.dist.time_series_metric_with_sklearn"
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
            }
        }
    }

    assert models_clustering == expected_models_clustering

    expected_models_cvi = {
        "config_CVI": {
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
        }
    }

    assert get_models_config(config_cvi) == expected_models_cvi

