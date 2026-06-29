from typing import Any, Sequence, Union
import inspect
import json
import numpy as np
import os
from pathlib import Path

import importlib

from .utils import write_json

CONFIG_DATA_BASE = {
    "config_data" : {
        "path_data" : "./data/",
        "path_res" : "./res/",
        "max_n_samples" : 10000,
        "max_n_labels" : 20,
        "max_n_dims" : None,
        "exclude" : [],
        "include_only" : [],
    }
}

CONFIG_CLUSTERING_BASE = {
    "config_clustering" : {
        "VI_max" : 0.2,
        "seed" : 221,
        "k_range" : [1, 25],
        "KMeans" : {
            "model" : "sklearn.cluster.KMeans",
            "model_kw" : {},
            "fit_predict_kw" : {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "Agglomerative-Ward" : {
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "ward",
                "metric": "euclidean"
            },
            "fit_predict_kw": {},
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
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "SpectralClustering" : {
            "model": "sklearn.cluster.SpectralClustering",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KMedoids" : {
            "model": "kmedoids.KMedoids",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        }
    }
}

CONFIG_CLUSTERING_TIME_SERIES_BASE = {
    "config_clustering" : {
        "VI_max" : 0.2,
        "seed" : 221,
        "k_range" : [1, 25],
        "KASBA" : {
            "model" : "aeon.clustering.KASBA",
            "model_kw" : {},
            "fit_predict_kw" : {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KShape" : {
            "model": "aeon.clustering.KShape",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKMeans" : {
            "model": "aeon.clustering.TimeSeriesKMeans",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKMedoids" : {
            "model": "aeon.clustering.TimeSeriesKMedoids",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKernelKMeans" : {
            "model": "aeon.clustering.TimeSeriesKernelKMeans",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesCLARA" : {
            "model": "aeon.clustering.TimeSeriesCLARA",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesCLARANS" : {
            "model": "aeon.clustering.TimeSeriesCLARANS",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "ElasticSOM" : {
            "model": "aeon.clustering.ElasticSOM",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KSpectralCentroid" : {
            "model": "aeon.clustering.KSpectralCentroid",
            "model_kw": {},
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "Agglomerative-Single" : {
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

CONFIG_CVI_BASE = {
    "config_CVI" : {
        "seed" : 221,
        "Hartigan" : {
            "cvi" : "pycvi.cvi.Hartigan"
        },
        "CalinskiHarabasz" : {
            "cvi" : "pycvi.cvi.CalinskiHarabasz"
        },
        "GapStatistic" : {
            "cvi" : "pycvi.cvi.GapStatistic"
        },
        "Silhouette" : {
            "cvi" : "pycvi.cvi.Silhouette"
        },
        "ScoreFunction" : {
            "cvi" : "pycvi.cvi.ScoreFunction"
        },
        "MaulikBandyopadhyay" : {
            "cvi" : "pycvi.cvi.MaulikBandyopadhyay"
        },
        "SD" : {
            "cvi" : "pycvi.cvi.SD"
        },
        "SDbw" : {
            "cvi" : "pycvi.cvi.SDbw"
        },
        "Dunn" : {
            "cvi" : "pycvi.cvi.Dunn"
        },
        "XB" : {
            "cvi" : "pycvi.cvi.XB"
        },
        "XBStar" : {
            "cvi" : "pycvi.cvi.XBStar"
        },
        "DB" : {
            "cvi" : "pycvi.cvi.DB"
        },
        "Inertia-sum" : {
            "cvi" : "pycvi.cvi.Inertia",
            "cvi_kw" : {
                "reduction" : "sum"
            }
        },
        "Diameter-max" : {
            "cvi" : "pycvi.cvi.Diameter",
            "cvi_kw" : {
                "reduction" : "max"
            }
        }
    }
}

CONFIG_DEFAULT_VALUES = {
    "config_data" : {
        "path_data" : "",
        "path_res" : "",
        "max_n_samples" : None,
        "max_n_labels" : None,
        "max_n_dims" : None,
        "exclude" : [],
        "include_only" : [],
    },
    "config_clustering" : {
        "VI_max" : None,
        "seed" : 221,
        "lower" : {
            "model_kw" : {},
            "fit_predict_kw" : {},
            "scaler" : None,
            "scaler_kw" : {}
        }
    },
    "config_CVI" : {
        "seed" : 221,
        "lower" : {
            "cvi_kw" : {}
        }
    }
}

def make_default_config(
    filenames: Union[str, Sequence[str]] = [
        "config-data.json", "config-clustering.json",
        "config-clustering-time_series.json", "config-CVI.json"
    ]
) -> None:
    """
    Create default config files in the given path.

    Parameters
    ----------
    filenames : Union[str, Sequence[str]], optional
        Filenames for the config files to be created. If a single string
        is provided, it will be used as the path to the directory where
        the config files will be created. If a sequence of strings is
        provided, must contain 3 or 4 string (a 4th one for the case of
        time series clustering), and each string will be used as
        filename to create the config files in multiple directories, by
        default ``[ "config-data.json", "config-clustering.json",
        "config-clustering-time_series.json", "config-CVI.json" ]``.
    """


    default_filenames = [
        "config-data.json", "config-clustering.json",
        "config-clustering-time_series.json", "config-CVI.json"
    ]
    if isinstance(filenames, str):
        filenames = [filenames + "/" + fname for fname in default_filenames]
    if isinstance(filenames, Sequence) and len(filenames) == 1:
        filenames = [filenames[0] + "/" + fname for fname in default_filenames]

    # Make sure all path exists otherwise create it
    for fname in filenames:
        p = Path(fname)
        p.parent.mkdir(parents=True, exist_ok=True)

    write_json(filenames[0], CONFIG_DATA_BASE)
    write_json(filenames[1], CONFIG_CLUSTERING_BASE)
    if len(filenames) == 4:
        write_json(filenames[2], CONFIG_CLUSTERING_TIME_SERIES_BASE)
    write_json(filenames[-1], CONFIG_CVI_BASE)

def get_obj_from_string(obj_str: str) -> Any:
    """
    Return the object corresponding to a string representing it.

    The object can be a class or a function. The string must be of the
    form `"package.module.class"` or `"package.module.function"`.

    The corresponding class or function will be imported.

    If the string is not of the correct form, an ImportError will be
    raised.

    Parameters
    ----------
    obj_str : str
        String representing the object to be imported.

    Returns
    -------
    Any
        The object corresponding to the string.
    """
    # rsplit(".", 1) will split once, at the very last occurence
    module_path, obj_name = obj_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def serialize(obj):
    # To check if the object is serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, Sequence):
            res = [serialize(x) for x in obj]
        # If we are dealing with a dict of object
        elif isinstance(obj, dict):
            res = {key : serialize(item) for key, item in obj.items()}
        # If we are dealing with a dict of object
        elif isinstance(obj, np.ndarray):
            res = obj.tolist()
        else:
            # True for classes/types, False for instances
            if inspect.isclass(obj):
                res = class_to_string(obj)
            elif inspect.isfunction(obj):
                res = f"{obj.__module__}.{obj.__name__}"
            else:
                res = obj_to_string(obj)
        return res

def class_to_string(cls):
    return f"{cls.__module__}.{cls.__name__}"

def obj_to_string(obj):
    cls = obj.__class__
    return f"Instance of {cls.__module__}.{cls.__name__}"

def add_default(config:dict) -> dict:
    """
    Complement given config with default parameters

    For data: Add ``exclude`` and ``include``, ``max_n_samples``,
    ``max_n_labels``, ``max_n_dims``, ``path_data``, ``path_res`` if not
    present (so none are mandatory, but they are all recommended)

    For the general clustering config: Add ``VI_max``, ``seed``if not present (but not ``k_range``, which is in any case mandatory).

    For each clustering model: Add ``model_kw``, ``fit_predict_kw``,
    ``scaler``, ``scaler_kw`` if not present (but not ``model``,
    which is in any case mandatory).

    For the general CVI config: Add ``seed`` if not present.

    For each CVI model: Add ``cvi_kw`` if not present (but not ``cvi``,
    which is in any case mandatory).
    """

    # Get the subset of the config about the models (cvi, clustering)
    models_config = get_models_config(config)

    complete_dict = {}

    for config_type, default_values in CONFIG_DEFAULT_VALUES.items():

        # Don't try to add default values if the config type is not present
        if config_type in config:

            # Complete higher level config with default values, if not present
            complete_dict[config_type] = default_values | config[config_type]

            # Complete lower level config with default values, if not present
            # 1. Remove the "lower" key from the complete_dict
            has_lower = complete_dict[config_type].pop("lower", False)
            # 2. Add the lower level default values to each model's config
            if has_lower:
                for model, model_config in models_config[config_type].items():
                    complete_dict[config_type][model] = default_values["lower"] | model_config

    return complete_dict


def simplify_config_dict(config:dict) -> dict:
    """
    Translate a given dict to a jsonable dict"

    Make sure that classes and objects are written as package.module.class
    """
    #normal_types = [Sequence, str, list, dict, int, float, bool, type(None)]

    simpler_dict = {}
    for k, v in config.items():
        # Serialize value if necessary
        simpler_dict[k] = serialize(v)
    return simpler_dict

def interpret_saved_dict(config:dict) -> dict:
    """
    Translate a given dict to a config dict, using classes and functions"
    """

    interpreted_dict = {}
    for k, v in config.items():
        # If the value is a string, try to interpret it as a class or function
        if isinstance(v, str):
            try:
                interpreted_dict[k] = get_obj_from_string(v)
            except (ImportError, AttributeError):
                interpreted_dict[k] = v
        # If the value is a dict, recursively interpret it
        elif isinstance(v, dict):
            interpreted_dict[k] = interpret_saved_dict(v)
        # Else, assume that the format is correct and keep the value as is
        else:
            interpreted_dict[k] = v
    return interpreted_dict


def load_config_as_dict(config_fname: str) -> dict:
    """
    Load a given config (json file) as a dict"

    Make sure that classes and objects that are written as package.module.class,
    are then loaded properly, as objects and classes, not as strings
    """
    with open(config_fname, "r") as f:
        config_dict = json.load(f)
    return interpret_saved_dict(config_dict)


def get_mandatory_keys() -> dict:
    """return the mandatory keys of each config type"""

    keys = {
        "config_data" : {
            "mandatory" : [
                "path_data", "path_res", "max_n_samples", "max_n_labels",
                "max_n_dims", "exclude", "include_only",
            ],
            "lower" : None,
        },
        "config_clustering" : {
            "mandatory" : ["VI_max", "seed", "k_range"],
            "lower" : [
                "model", "model_kw", "fit_predict_kw",
                "scaler", "scaler_kw",
            ],
        },
        "config_CVI" : {
            "mandatory" : ["seed"],
            "lower" : ["cvi", "cvi_kw"],
        },
    }

    return keys

def get_models_config(config:dict) -> dict:
    """
    Extract models config from a config (cvi, clustering)
    """
    all_keys = get_mandatory_keys()

    model_config = {}

    for config_type, config_keys in all_keys.items():

        # Get keys that are not mandatory (and thus model keys)
        # And extract the config of each model (clustering or cvi)
        model_config[config_type] = {
            k : config[config_type][k] for k in config[config_type]
            if k not in all_keys[config_type]["mandatory"]
        }

    return model_config


def check_config(config:dict) -> bool:
    """
    Check that the config has all necessary keys (after adding default).

    This function assumes that default parameters have already been added
    to the user-defined config
    """

    all_keys = get_mandatory_keys()

    # Subset of the config that is only about the models (cvi, clustering)
    model_config = get_models_config(config)


    for config_type, config_keys in all_keys.items():

        # If the config is present, then it must follow requirements
        if config_type in config:

            # ============== High level mandatory keys ==============
            # Message to show
            msg = f"Configuration {config_type} missing mandatory keys, got {list(config[config_type].keys())}, expected {config_keys["mandatory"]}"

            # Test that mandatory keys are here for each config type
            assert all(
                [k in config[config_type] for k in config_keys["mandatory"]]
            ), msg

            # ============== Lower level mandatory keys ==============

            # Go to next config type if there is no lower key to check
            # Typically there is no lower key for data
            if all_keys[config_type]["lower"] is None:
                continue

            # Check that there is at least one model
            if not model_config[config_type]:
                raise ValueError(f"No individual models configured in {config_type}.")

            # Check each models (clustering or cvi) one by one
            mandatory_keys = all_keys[config_type]["lower"]

            for model, config_model in model_config[config_type].items():

                msg = f"Model configuration missing mandatory key in {config_type}. Got {list(config_model.keys())}, expected {mandatory_keys}."

                # They all have necessary sub-keys
                assert all(
                    [k in config_model for k in mandatory_keys]
                ), msg

    return True
