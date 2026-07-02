
import numpy as np
import pytest


from pycvi.cluster import get_clustering

from clusterexp.clustering import (
    decompose_exp_fnames, f_quality, compute_VI_quality
)

def test_f_quality():
    # Test that f_quality returns the expected values for known VI values
    quality = f_quality(0.0)
    assert np.isclose(quality, 1.0)
    assert isinstance(quality, float)

def test_compute_VI_quality():
    # Test that compute_VI_quality returns the expected values for known true and predicted clusterings
    true_clusters = get_clustering(np.array([0, 0, 1, 1]))
    clusterings = {
        2: get_clustering(np.array([0, 0, 1, 1])),
        3: get_clustering(np.array([0, 0, 1, 2])),
        4: get_clustering(np.array([0, 1, 2, 3]))
    }
    VIs, qualities = compute_VI_quality(true_clusters, clusterings)
    assert isinstance(VIs, dict)
    assert isinstance(qualities, dict)
    assert all(isinstance(v, float) for v in VIs.values())
    assert all(isinstance(q, float) for q in qualities.values())

def test_decompose_exp_fnames():
    exp_fnames = [
        "res/KMeans/artificial/2d-4c-no-clustering.json",
        "res/Agglomerative-Single/artificial/2d-4c-no-clustering.json",
        "res/Agglomerative-Ward/artificial/2d-4c-no-clustering.json",
        "res/KMeans/artificial/2d-4c-no-CVI.json",
        "res/Agglomerative-Single/artificial/2d-4c-no-CVI.json",
        "res/Agglomerative-Ward/artificial/2d-4c-no-CVI.json"
    ]
    path_res = "res/"
    expected_output = [
        ("KMeans", "artificial/2d-4c-no"),
        ("Agglomerative-Single", "artificial/2d-4c-no"),
        ("Agglomerative-Ward", "artificial/2d-4c-no"),
        ("KMeans", "artificial/2d-4c-no"),
        ("Agglomerative-Single", "artificial/2d-4c-no"),
        ("Agglomerative-Ward", "artificial/2d-4c-no")
    ]
    output = decompose_exp_fnames(exp_fnames, path_res)
    assert output == expected_output