import numpy as np
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from .data import save_data_labels


DataArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]

def showcase(
    path_data: str = "./datasets/sklearn_showcase",
    N: int = 500,
    seed: int = 30,
    standardise: bool = True
) -> list[tuple[DataArray, LabelArray]]:
    """
    Generate a small collection of synthetic clustering datasets.

    The generated datasets mirror the examples used in scikit-learn's cluster
    comparison gallery and can optionally be standardized and written to disk.

    Courtesy to https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html


    Parameters
    ----------
    path_data : str, default="./datasets/sklearn_showcase"
        Output directory where each generated dataset should be saved. If an
        empty string evaluates to ``False``, files are not written.
    N : int, default=500
        Number of samples to generate for each dataset.
    seed : int, default=30
        Seed used for NumPy's random number generator.
    standardise : bool, default=True
        Whether to standardize each dataset before saving and returning it.

    Returns
    -------
    list[tuple[DataArray, LabelArray]]
        List of ``(data, labels)`` pairs for the generated datasets, in the
        order noisy circles, noisy moons, varied blobs, anisotropic blobs,
        isotropic blobs, and no-structure noise.
    """

    np.random.seed(seed)

    # Generate the datasets
    noisy_circles = datasets.make_circles(n_samples=N, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=N, noise=0.05)
    blobs = datasets.make_blobs(n_samples=N, random_state=8)
    no_structure = np.random.rand(N, 2), np.zeros(N, )

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=N, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=N, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    list_datasets = [
        ( noisy_circles, "noisy_circles", ),
        ( noisy_moons, "noisy_moons", ),
        ( varied, "varied", ),
        ( aniso, "aniso", ),
        ( blobs, "blobs", ),
        ( no_structure, "no_structure", ),
    ]

    res = []

    for dataset, name in list_datasets:

        data, labels = dataset

        # normalize dataset for easier parameter selection
        if standardise:
            data = StandardScaler().fit_transform(data)

        if path_data:
            save_data_labels(data, labels, f"{path_data}/{name}")

        res.append((data, labels))

    return res









