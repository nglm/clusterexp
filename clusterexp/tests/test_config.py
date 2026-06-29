import sklearn
import numpy
import os
import pytest

from ..config import (
    CONFIG_DATA_BASE, CONFIG_CLUSTERING_BASE,
    CONFIG_CLUSTERING_TIME_SERIES_BASE, CONFIG_CVI_BASE, CONFIG_DEFAULT_VALUES,
    make_default_config, simplify_config_dict)
from ..utils import write_json

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

def test_simplify_config_dict():
    simpler_dict = simplify_config_dict(config_1)
    write_json("test_simplify_dict.json", simpler_dict)


def test_constants():
    assert isinstance(CONFIG_DATA_BASE, dict)
    assert isinstance(CONFIG_CLUSTERING_BASE, dict)
    assert isinstance(CONFIG_CLUSTERING_TIME_SERIES_BASE, dict)
    assert isinstance(CONFIG_CVI_BASE, dict)
    assert isinstance(CONFIG_DEFAULT_VALUES, dict)

def test_make_default_config():
    make_default_config("test_make_default_config")
    assert os.path.isfile("test_make_default_config/config-data.json")
    assert os.path.isfile("test_make_default_config/config-clustering.json")
    assert os.path.isfile("test_make_default_config/config-clustering-time_series.json")
    assert os.path.isfile("test_make_default_config/config-CVI.json")
