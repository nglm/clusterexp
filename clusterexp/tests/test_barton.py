
import numpy as np
import pytest

from clusterexp.barton import (
    get_list_datasets_from_github, get_data_labels, arff_from_github,
    load_data_from_github, URL_ROOT
)

def test_get_list_datasets_from_github():

    # With all datasets
    datasets = get_list_datasets_from_github(
        data_source="artificial", with_invalid=True, with_unknown_k=True
    )
    assert isinstance(datasets, list)
    assert len(datasets) == 122
    assert all(isinstance(d, str) for d in datasets)

    # With only valid datasets
    datasets = get_list_datasets_from_github(
        data_source="artificial", with_invalid=False, with_unknown_k=False
    )
    assert isinstance(datasets, list)
    assert len(datasets) == 117

def test_arff_from_github():
    # Test with a known dataset
    fname = "artificial/long3.arff"
    data, meta = arff_from_github(f"{URL_ROOT}{fname}")
    assert isinstance(data, np.ndarray)

def test_load_data_from_github():
    # Test with a known dataset
    fname = "artificial/long3.arff"
    data, labels, meta = load_data_from_github(
        f"{URL_ROOT}{fname}", with_labels=True
    )
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(data) == len(labels)

def test_get_data_labels():

    # A normal dataset with labels
    fname = "long3.arff"
    data, labels, meta = get_data_labels(
        fname=fname, url=f"{URL_ROOT}artificial/"
    )
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == 2

    # A dataset without labels but actually unimodal
    fname = "birch-rg1.arff"
    data, labels, meta = get_data_labels(
        fname=fname, url=f"{URL_ROOT}artificial/"
    )
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(np.unique(labels)) == 1

    # A dataset without labels and not unimodal
    fname = "birch-rg3.arff"
    data, labels, meta = get_data_labels(
        fname=fname, url=f"{URL_ROOT}artificial/"
    )
    assert isinstance(data, np.ndarray)
    assert labels is None