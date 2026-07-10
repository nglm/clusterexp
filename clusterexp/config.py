"""
Module for configuration of scripts (data, clustering, CVI)
"""

from typing import Any, Sequence, Union
import numpy as np
from pathlib import Path

from .utils import write_json, interpret_dict

CONFIG_DATA_BASE = {
    "config_data" : {
        "path_data" : "./data/",
        "path_res" : "./res/",
        "max_n_samples" : 10000,
        "max_n_labels" : 20,
        "max_n_dims" : float('inf'),
        "exclude" : [],
        "include_only" : [],
    }
}

CONFIG_CLUSTERING_BASE = {
    "config_clustering" : {
        "k_range" : [1, 25],
        "KMeans" : {
            "model" : "sklearn.cluster.KMeans",
            "model_kw" : {
                "random_state" : 221
            },
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
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KMedoids" : {
            "model": "kmedoids.KMedoids",
            "model_kw": {
                "metric": "euclidean",
                "random_state" : 221,
                },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        }
    }
}

CONFIG_CLUSTERING_TIME_SERIES_BASE = {
    "config_clustering" : {
        "k_range" : [1, 25],
        "KASBA" : {
            "model" : "aeon.clustering.KASBA",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw" : {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KShape" : {
            "model": "aeon.clustering.KShape",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKMeans" : {
            "model": "aeon.clustering.TimeSeriesKMeans",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKMedoids" : {
            "model": "aeon.clustering.TimeSeriesKMedoids",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKernelKMeans" : {
            "model": "aeon.clustering.TimeSeriesKernelKMeans",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesCLARA" : {
            "model": "aeon.clustering.TimeSeriesCLARA",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesCLARANS" : {
            "model": "aeon.clustering.TimeSeriesCLARANS",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "ElasticSOM" : {
            "model": "aeon.clustering.ElasticSOM",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "KSpectralCentroid" : {
            "model": "aeon.clustering.KSpectralCentroid",
            "model_kw" : {
                "random_state" : 221
            },
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
        "quality_true_min" : 0.,
        "quality_best_min" : 0.,
        "best_q_true_only" : False,
        "best_q_max_only" : False,
        "Hartigan" : {
            "cvi" : "pycvi.cvi.Hartigan",
            "cvi_kw" : {
                "rng" : 221
            }
        },
        "CalinskiHarabasz" : {
            "cvi" : "pycvi.cvi.CalinskiHarabasz",
            "cvi_kw" : {
                "rng" : 221
            }
        },
        "GapStatistic" : {
            "cvi" : "pycvi.cvi.GapStatistic",
            "cvi_kw" : {
                "rng" : 221
            }
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
            "cvi_init_kw" : {
                "reduction" : "sum"
            }
        },
        "Diameter-max" : {
            "cvi" : "pycvi.cvi.Diameter",
            "cvi_init_kw" : {
                "reduction" : "max"
            }
        }
    }
}

CONFIG_DEFAULT_VALUES = {
    "config_data" : {
        "path_data" : "",
        "path_res" : "",
        "max_n_samples" : float('inf'),
        "max_n_labels" : float('inf'),
        "max_n_dims" : float('inf'),
        "exclude" : [],
        "include_only" : [],
    },
    "config_clustering" : {
        "k_range" : None,
        "lower" : {
            "model_kw" : {},
            "fit_predict_kw" : {},
            "scaler" : None,
            "scaler_kw" : {}
        }
    },
    "config_CVI" : {
        "seed" : 221,
        "quality_true_min" : 0.0,
        "quality_best_min" : 0.0,
        "lower" : {
            "cvi_init_kw" : {},
            "cvi_kw" : {}
        }
    }
}

def get_mandatory_keys() -> dict:
    """
    Return the mandatory keys and expected value types for configs.

    Returns
    -------
    dict
        Mapping from top-level config sections to their required keys
        and, when relevant, required per-model keys.
    """

    keys = {
        "config_data" : {
            "mandatory" : {
                "path_data": str,
                "path_res": str,
                "max_n_samples": (int, float),
                "max_n_labels": (int, float),
                "max_n_dims": (int, float),
                "exclude": list,
                "include_only": list
            },
            "lower" : None,
        },
        "config_clustering" : {
            "mandatory" : {
                "k_range": (list, tuple, np.ndarray),
            },
            "lower" : {
                # Interpreted: object, but saved: str
                "model" : object,
                "model_kw": dict,
                "fit_predict_kw": dict,
                # Interpreted: object or None, but saved: str or NoneType
                "scaler": object,
                "scaler_kw": dict,
            },
        },
        "config_CVI" : {
            "mandatory" : {
                "seed": int,
                "quality_true_min": (int, float),
                "quality_best_min": (int, float),
                "best_q_true_only": bool,
                "best_q_max_only": bool,
            },
            "lower" : {
                "cvi": object,
                "cvi_init_kw": dict,
                "cvi_kw": dict
            },
        },
    }

    return keys

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

    Returns
    -------
    None
        This function writes default configuration files to disk.
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


def add_default(config:dict) -> dict:
    """
    Complement a config dictionary with default parameters.

    For data: Add ``exclude`` and ``include``, ``max_n_samples``,
    ``max_n_labels``, ``max_n_dims``, ``path_data``, ``path_res`` if not
    present (so none are mandatory, but they are all recommended)

    For the general clustering config: Add  ``seed``if not present (but not ``k_range``, which is in any case mandatory).

    For each clustering model: Add ``model_kw``, ``fit_predict_kw``,
    ``scaler``, ``scaler_kw`` if not present (but not ``model``,
    which is in any case mandatory).

    For the general CVI config: Add ``quality_true_min``, ``quality_best_min``, ``seed``, ``best_q_true_only``, ``best_q_max_only`` if not present.

    For each CVI model: Add ``cvi_kw`` and ``cvi_init_kw`` if not
    present (but not ``cvi``, which is in any case mandatory).

    Parameters
    ----------
    config : dict
        Partially specified configuration dictionary.

    Returns
    -------
    dict
        Configuration dictionary completed with missing default values.
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



def interpret_config(config:Union[dict, str]) -> dict:
    """
    Interpret a config dictionary or JSON file into proper config dict.

    Make sure that classes and functions are written as
    package.module.class

    This function will also add the default values to the config.

    For a function that works for dict in general that are not config,
    see :func:`interpret_dict` .

    Parameters
    ----------
    config : Union[dict, str]
        Config dictionary or path to a JSON config file.

    Returns
    -------
    dict
        Interpreted config with import strings resolved and default
        values added when top-level config sections are present.
    """

    interpreted_dict = interpret_dict(config)

    # Check if we are in the top level case (this function is recursive)
    top_level_keys = {"config_data", "config_clustering", "config_CVI"}
    # Check that intersection of subsets is not empty
    if interpreted_dict.keys() & top_level_keys:
        # Then it's time to add default values to the config
        interpreted_dict = add_default(interpreted_dict)
    return interpreted_dict

def get_models_config(config:dict) -> dict:
    """
    Extract per-model configuration blocks from a config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.

    Returns
    -------
    dict
        Nested dictionary containing only the model
        entries for each supported config section.
    """
    all_keys = get_mandatory_keys()

    model_config = {}

    for config_type, config_keys in all_keys.items():

        if config_type not in config:
            # Skip this config type if it is not present in the given config
            continue

        # Get keys that are not mandatory (and thus model keys)
        # And extract the config of each model (clustering or cvi)
        model_config[config_type] = {
            k : config[config_type][k] for k in config[config_type]
            if k not in all_keys[config_type]["mandatory"]
        }

    return model_config


def check_config( config:dict, ) -> bool:
    """
    Check that the config has all necessary keys (after adding default).

    This function assumes that default parameters have already been added
    to the user-defined config.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.

    Returns
    -------
    bool
        ``True`` when all mandatory sections and keys are present.
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
