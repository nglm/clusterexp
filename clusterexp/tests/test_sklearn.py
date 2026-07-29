import pytest

import numpy as np

from clusterexp.sklearn import showcase

def test_showcase():

    # Original dataset but not standardized
    path_data = "test/datasets/sklearn_showcase"
    datasets = showcase(f"{path_data}_no_std", standardise=False)

    assert type(datasets) == list
    assert type(datasets[0]) == tuple
    assert type(datasets[0][0]) == np.ndarray
    assert type(datasets[0][1]) == np.ndarray
    assert len(datasets) == 6

    # Original dataset
    showcase(f"{path_data}")

    # More datapoints
    showcase(f"{path_data}_5000")
