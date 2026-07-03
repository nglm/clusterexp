
import numpy as np
import pytest

from ..barton import load_data_from_github, URL_ROOT

from clusterexp.data import (
    find_datasets, load_data_labels, filter_datasets, process_labels
)

path_data = "./example_data/"

def test_find_datasets():
    datasets = find_datasets(path_data)

    # Check that we get a matching number of datasets
    assert isinstance(datasets, list)
    assert len(datasets) == 11

    # Check that the path is included
    assert all("artificial/" in d or "real-world/" in d for d in datasets)

def test_load_data_labels():
    datasets = find_datasets(path_data)
    for d in datasets:
        data, labels = load_data_labels(f"{d}")

        # Check types
        assert isinstance(data, np.ndarray)
        assert isinstance(labels, np.ndarray)

        # Check that the number of samples in data and labels match
        assert len(data) == len(labels)

def test_filter_datasets():
    constraints1 = {
        "max_n_samples": 1000,
        "max_n_labels": 10,
        "max_n_dims": 10,
        "exclude": ["arrhythmia", "2d-4c-no"],
        "include_only": ["artificial/"]
    }
    datasets = find_datasets(path_data)

    constraints2 = {}

    filtered1 = filter_datasets(datasets, **constraints1)
    filtered2 = filter_datasets(datasets, **constraints2)

    for filtered in [filtered1, filtered2]:

        # Check types
        assert isinstance(filtered, dict)
        assert "kept_datasets" in filtered
        assert "dropped_datasets" in filtered
        assert isinstance(filtered["kept_datasets"], list)
        assert isinstance(filtered["dropped_datasets"], dict)

        # Check that all the original datasets end up in one of the categories
        all_datasets = set(
            filtered["kept_datasets"]
            + sum(filtered["dropped_datasets"].values(), [])
        )

        assert set(datasets) == all_datasets

    # Check that there is no "real-world" in the kept datasets for filtered1
    assert all("real-world/" not in d for d in filtered1["kept_datasets"])
    # Check that there is real-world in the excluded datasets
    assert all("real-world/" in d for d in filtered1["dropped_datasets"]["include_only"])

    # Excluded datasets should be in the dropped datasets for filtered1
    assert all(["arrhythmia" not in d for d in filtered1["kept_datasets"]])
    assert all(["2d-4c-no" not in d for d in filtered1["kept_datasets"]])
    assert any("arrhythmia" in d for d in filtered1["dropped_datasets"]["exclude"])
    assert any("2d-4c-no" in d for d in filtered1["dropped_datasets"]["exclude"])

    # Check that there is no dropped datasets for filtered2
    assert len(filtered2["kept_datasets"]) == len(datasets)
    assert set(filtered2["kept_datasets"]) == set(datasets)
    assert filtered2["dropped_datasets"] == {
        "max_n_samples": [],
        "max_n_labels": [],
        "max_n_dims": [],
        "exclude": [],
        "include_only": [],
    }


def test_process_labels():
    fname = "artificial/long3.arff"
    data, labels, meta = load_data_from_github(
        f"{URL_ROOT}{fname}", with_labels=True
    )

    processed_labels, n_labels = process_labels(labels)
    assert isinstance(processed_labels, np.ndarray)
    assert n_labels == 2