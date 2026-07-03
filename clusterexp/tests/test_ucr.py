
import numpy as np
import pytest

from clusterexp.ucr import (
    find_datasets_UCR, save_data_labels_UCR, get_data_labels_UCR, ILL_FORMATED
)

from pathlib import Path

home_dir = Path.home()

PATH_UCR_LOCAL = f"example_data/UCR/"

def test_find_datasets_UCR():

    # With all datasets
    datasets = find_datasets_UCR(
        path_ucr=PATH_UCR_LOCAL, with_ill_formated=True
    )
    assert isinstance(datasets, list)
    assert all(isinstance(d, str) for d in datasets)
    assert len(datasets) == 7

    # With only valid datasets
    datasets = find_datasets_UCR(
        path_ucr=PATH_UCR_LOCAL, with_ill_formated=False
    )
    assert isinstance(datasets, list)
    assert all(isinstance(d, str) for d in datasets)
    assert len(datasets) == 4
    for d in datasets:
        if any(ill in d for ill in ILL_FORMATED):
            assert "Missing_value_and_variable_length_datasets_adjusted" in d

def test_get_data_labels_UCR():

    # A normal dataset with labels
    fname = "BeetleFly/BeetleFly"
    data, labels = get_data_labels_UCR(
        fname=f"{PATH_UCR_LOCAL}{fname}_TRAIN.tsv"
    )
    assert isinstance(data, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(data) == len(labels)
    assert data.shape[2] == 1
    assert len(data.shape) == 3
