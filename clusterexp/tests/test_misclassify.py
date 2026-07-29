import pytest

import numpy as np
from sklearn.datasets import make_blobs

from clusterexp.misclassify import (
    full_random, balanced, bully, subclustering, superclustering,
    flag_misclassified, set_misclassified,
    stats,
)

def test_set_misclassified():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for r in [0.1, [0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]]:
        for apply in [True, False]:
            new_r = set_misclassified(
                misclassified=r, y=y, apply_misclassification_to_all_clusters=apply
            )

            assert isinstance(new_r, list)
            assert len(new_r) == 5
            assert all(isinstance(x, float) for x in new_r)

def test_stats():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for r in [0.1, [0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5]]:
        for apply in [True, False]:
            y_wrong = full_random(
                X, y, misclassified=r, apply_misclassification_to_all_clusters=apply
            )

            stats_dict = stats(y, y_wrong)

            assert isinstance(stats_dict, dict)


def test_full_random():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for r in [0.1, [0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4, 0.5]]:
        for apply in [True, False]:
            for is_global in [True, False]:
                y_wrong = full_random(
                    X, y, misclassified=r, apply_misclassification_to_all_clusters=apply,
                    global_misclassification=is_global
                )

                assert isinstance(y_wrong, np.ndarray)
                assert y_wrong.shape == y.shape


def test_balanced():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = balanced(X, y)

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

    y_wrong = balanced(X, y, allow_same_closest=True)

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

    # Artificially decrease one cluster size to trigger the error
    # 50% of the point of class 0 are relabeled to 1
    y_error = np.copy(y)
    idx_y_0 = np.where(y == 0)[0]
    idx_y_0_to_change = idx_y_0[:len(idx_y_0)//2]
    y_error[idx_y_0_to_change] = 1

    y_wrong = balanced(
        X, y_error, misclassified=0.8,
    )

def test_bully():

    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)
    counts_y = np.bincount(y)


    y_wrong = bully(X, y)

    # There is no cluster fully eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    assert len(np.unique(y_wrong)) == 5


    y_wrong = bully(X, y, misclassified=0.5)

    # Two clusters are fully eaten, and one is half eaten
    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape
    counts_y_wrong = np.bincount(y_wrong)
    print(counts_y, counts_y_wrong)
    assert len(np.unique(y_wrong)) < len(np.unique(y))
    assert len(np.unique(y_wrong)) == 3

def test_subclustering():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    for method in ["grouped", "random"]:
        for misclassified in [0.1, [0.1, 0.2, 0.3]]:
            for misclassify_minority in [True, False]:
                for apply_misclassification_to_all_clusters in [True, False]:
                    y_wrong = subclustering(
                        X, y, method=method, misclassified=misclassified,
                        misclassify_minority=misclassify_minority,
                        apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
                    )

                    assert isinstance(y_wrong, np.ndarray)
                    assert y_wrong.shape == y.shape
                    assert len(np.unique(y_wrong)) > len(np.unique(y))

                    # Check the new number of clusters
                    if isinstance(misclassified, list):
                        assert len(np.unique(y_wrong)) == len(np.unique(y)) + len(misclassified)
                    elif isinstance(misclassified, float) and not apply_misclassification_to_all_clusters:
                        assert len(np.unique(y_wrong)) == len(np.unique(y)) + 1
                    elif isinstance(misclassified, float) and apply_misclassification_to_all_clusters:
                        assert len(np.unique(y_wrong)) == 2*len(np.unique(y))

def test_superclustering_agglomerative():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    n_classes = len(np.unique(y))

    for misclassified in [0., 0.39, 0.81, 1]:
        for is_upper_bound in [True, False]:

            y_wrong = superclustering(
                X,
                y,
                misclassified=misclassified,
                method="agglomerative",
                is_upper_bound=is_upper_bound,
            )

            assert isinstance(y_wrong, np.ndarray)
            assert y_wrong.shape == y.shape
            n_classes_wrong = len(np.unique(y_wrong))
            assert n_classes_wrong <= n_classes
            assert n_classes_wrong >= 1

            # There is always at least one class merged in that case
            if not is_upper_bound:
                assert n_classes_wrong < n_classes

            if misclassified == 0.39 and is_upper_bound:
                assert n_classes_wrong == 4

            if misclassified >= 0.81:
                assert n_classes_wrong == 1


def test_superclustering_smallest():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    n_classes = len(np.unique(y))

    for misclassified in [0.0, 0.39, 0.81, 1.0]:
        for is_upper_bound in [True, False]:
            y_wrong = superclustering(
                X,
                y,
                misclassified=misclassified,
                method="smallest",
                is_upper_bound=is_upper_bound,
            )

            assert isinstance(y_wrong, np.ndarray)
            assert y_wrong.shape == y.shape
            n_classes_wrong = len(np.unique(y_wrong))
            assert n_classes_wrong <= n_classes
            assert n_classes_wrong >= 1

            if not is_upper_bound:
                assert n_classes_wrong < n_classes


def test_superclustering_mapping():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    n_classes = len(np.unique(y))

    superclusters = [[0, 1]]
    N_classes_disappeared = sum(len(l) - 1 for l in superclusters)

    y_wrong = superclustering(
        X,
        y,
        superclusters=superclusters,
        method="smallest",
    )

    n_classes_wrong = len(np.unique(y_wrong))

    assert isinstance(y_wrong, np.ndarray)
    assert y_wrong.shape == y.shape

    assert n_classes_wrong == n_classes - N_classes_disappeared


def test_flag_misclassified():
    X, y = make_blobs(n_samples=100, centers=5, n_features=2, random_state=42)

    y_wrong = full_random(X, y)

    # Check that the misclassified points are flagged correctly
    y_flagged = flag_misclassified(y, y_wrong)
    assert isinstance(y_flagged, np.ndarray)
    assert y_flagged.shape == y.shape
    assert np.sum(y_flagged == -1) == np.sum(y != y_wrong)

