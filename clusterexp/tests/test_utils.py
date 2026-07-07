import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy
import numpy as np
import os
import pytest

import pycvi
from pycvi.cvi import Hartigan

from ..utils import (
    class_to_string, get_obj_from_string, obj_to_string, write_json,
    simplify_dict, interpret_dict, load_json, extract_log_from_text,
)
from ..config import (
    make_default_config
)

config_1 = {
    "quality_true_min": 0.6,
    "quality_best_min": 0.6,
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





def test_class_to_string():

    # Testing with short class names
    cls1 = StandardScaler
    cls2 = KMeans
    s1 = class_to_string(cls1)
    s2 = class_to_string(cls2)
    assert s1 == "sklearn.preprocessing._data.StandardScaler"
    assert s2 == "sklearn.cluster._kmeans.KMeans"
    # Testing with long class names
    cls1 = sklearn.preprocessing.StandardScaler
    cls2 = sklearn.cluster.KMeans
    s1 = class_to_string(cls1)
    s2 = class_to_string(cls2)
    assert s1 == "sklearn.preprocessing._data.StandardScaler"
    assert s2 == "sklearn.cluster._kmeans.KMeans"

def test_get_obj_from_string():
    # Testing with preferred module names (non-hidden)
    s1 = "sklearn.preprocessing.StandardScaler"
    s2 = "sklearn.cluster.KMeans"
    obj1 = get_obj_from_string(s1)
    obj2 = get_obj_from_string(s2)
    assert obj1 == sklearn.preprocessing.StandardScaler
    assert isinstance(obj1(), sklearn.preprocessing.StandardScaler)
    assert isinstance(obj1(), StandardScaler)
    assert obj2 == sklearn.cluster.KMeans
    assert isinstance(obj2(), sklearn.cluster.KMeans)
    assert isinstance(obj2(), KMeans)

    # Testing with hidden module names
    s1 = "sklearn.preprocessing._data.StandardScaler"
    s2 = "sklearn.cluster._kmeans.KMeans"
    obj1 = get_obj_from_string(s1)
    obj2 = get_obj_from_string(s2)
    assert obj1 == sklearn.preprocessing.StandardScaler
    assert isinstance(obj1(), sklearn.preprocessing.StandardScaler)
    assert isinstance(obj1(), StandardScaler)
    assert obj2 == sklearn.cluster.KMeans
    assert isinstance(obj2(), sklearn.cluster.KMeans)
    assert isinstance(obj2(), KMeans)

def test_obj_to_string():
    # Testing with short class names
    cls1 = StandardScaler
    cls2 = KMeans
    s1 = obj_to_string(cls1())
    s2 = obj_to_string(cls2())
    assert s1 == "Instance of sklearn.preprocessing._data.StandardScaler"
    assert s2 == "Instance of sklearn.cluster._kmeans.KMeans"
    # Testing with long class names
    cls1 = sklearn.preprocessing.StandardScaler
    cls2 = sklearn.cluster.KMeans
    s1 = obj_to_string(cls1())
    s2 = obj_to_string(cls2())
    assert s1 == "Instance of sklearn.preprocessing._data.StandardScaler"
    assert s2 == "Instance of sklearn.cluster._kmeans.KMeans"

def test_simplify_dict():
    simpler_dict = simplify_dict(config_1)
    write_json("test/test_simplify_dict.json", simpler_dict)

def test_interpret_dict():
    dir = "test/test_interpret_dict"
    make_default_config(f"{dir}")
    config_clustering = load_json(f"{dir}/config-clustering.json")
    config_CVI = load_json(f"{dir}/config-CVI.json")

    config_CVI_interpreted = interpret_dict(config_CVI)
    config_clustering_interpreted = interpret_dict(config_clustering)

    config_clustering_interpreted
    assert config_CVI_interpreted["config_CVI"]["Hartigan"]["cvi"] == pycvi.cvi.Hartigan
    assert config_CVI_interpreted["config_CVI"]["Hartigan"]["cvi"] == Hartigan
    assert config_clustering_interpreted["config_clustering"]["KMeans"]["model"] == sklearn.cluster.KMeans
    assert config_clustering_interpreted["config_clustering"]["KMeans"]["model"] == KMeans
