from typing import Union, List, Tuple, Dict

import numpy as np
from numpy.typing import NDArray
import random
from sklearn.cluster import AgglomerativeClustering


DataArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]

def set_misclassified(
    misclassified: Union[float, List[float]],
    y: LabelArray,
    apply_misclassification_to_all_clusters: bool = True,
) -> List[int]:
    """
    Set the number of misclassified samples per cluster.

    Parameters
    ----------
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

    Returns
    -------
    List[int]
        List of misclassified fractions for each cluster.

    Raises
    ------
    ValueError
        If the length of the misclassified list is greater than the
        number of clusters in ``y`` or if any value in the list is not
        between 0 and 1.
    TypeError
        If ``misclassified`` is not a float or a list of floats.
    """

    N_clusters = len(np.unique(y))

    if isinstance(misclassified, float):
        # All clusters get the same rate
        if apply_misclassification_to_all_clusters:
            misclassified = [misclassified] * N_clusters
        # Only the first cluster get a non null misclassification rate
        else:
            misclassified = [misclassified] + [0.]*(N_clusters-1)
    elif isinstance(misclassified, list):

        # Raise error if more rates than clusters
        if len(misclassified) > N_clusters:
            raise ValueError(
                f"Length of misclassified list ({len(misclassified)}) "
                f"greater than the number of clusters ({N_clusters})."
            )

        # Complete with zeros if needed
        elif len(misclassified) < N_clusters:
            misclassified = misclassified + [0.] * (N_clusters - len(misclassified))

    # Raise error if wrong type
    else:
        raise TypeError(
            f"misclassified must be a float or a list of floats, "
            f"got {type(misclassified)}."
        )
    if any([r < 0 or r > 1 for r in misclassified]):
        raise ValueError(
            f"All values in misclassified list must be between 0 and 1. "
            f"Got {misclassified}."
        )
    return misclassified

def compute_centroids(X: DataArray, y: LabelArray) -> DataArray:
    """
    Compute the centroids of clusters in a dataset.

    Assumes that labels are integers 0..k-1, where k is the number of clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        The data points, shape (n_samples, n_features).
    y : NDArray[np.int_]
        The cluster labels for each data point, shape (n_samples,).

    Returns
    -------
    NDArray[np.float64]
        The centroids of the clusters, shape (n_clusters, n_features).
    """
    unique_labels = np.unique(y)
    centroids = np.zeros((len(unique_labels), X.shape[1]))
    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = X[y == label]
        # Compute the centroid of the cluster
        centroids[int(label)] = np.mean(cluster_points, axis=0)
    return centroids

def stats(
    y: LabelArray,
    y_wrong: LabelArray
) -> dict:
    """
    Compute statistics on the misclassified samples between two label arrays.

    Parameters
    ----------
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    y_wrong : NDArray[np.int_]
        Predicted cluster labels encoded as integers.

    Returns
    -------
    dict
        Dictionary containing:
        - "N_total": Total number of samples.
        - "N_misclassified": Total number of misclassified samples.
        - "misclassified": Fraction of misclassified samples.
        - "N_misclassified_per_cluster": Dictionary with the number of misclassified samples per cluster.
        - "misclassified_per_cluster": Dictionary with the fraction of misclassified samples per cluster.
        - "new_clusters": Dictionary with the number of samples in new clusters (clusters present in y_wrong but not in y).
        - "missing_clusters": List of clusters present in y but not in y_wrong.
    """
    classes = np.unique(y)
    classes_y_wrong = np.unique(y_wrong)
    new_clusters = {
        c: np.sum(y_wrong == c) for c in classes_y_wrong if c not in classes
    }
    missing_clusters = [c for c in classes if c not in classes_y_wrong]

    N_total = len(y)
    N_misclassified = np.sum(y != y_wrong)
    fraction_misclassified = N_misclassified / N_total

    N_misclassified_c = {
        c: np.sum((y == c) & (y_wrong != c)) for c in classes
    }
    fraction_misclassified_c = {
        c : N_misclassified_c[c] / np.sum(y == c) for c in classes
    }

    stats_dict = {
        "N_total": N_total,
        "N_misclassified": N_misclassified,
        "misclassified": fraction_misclassified,
        "N_misclassified_per_cluster": N_misclassified_c,
        "misclassified_per_cluster": fraction_misclassified_c,
        "new_clusters": new_clusters,
        "missing_clusters" : missing_clusters,
    }

    return stats_dict

def full_random(
    X: DataArray,
    y: LabelArray,
    misclassified: Union[float, List[float]] = 0.10,
    seed: int = 42,
    apply_misclassification_to_all_clusters: bool = True,
    global_misclassification: bool = False,
) -> LabelArray:
    """
    Randomly mislabel a fraction of datapoints within some clusters.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset used only to infer the number of samples.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.
    global_misclassification : bool, default=False
        If ``True`` and if ``misclassified`` was a float, randomly
        re-assign labels independently of the cluster. The result will
        be very similar to applying the same misclassification rate to
        all clusters, except that there will be small variations in the
        precise number of misclassified datapoint within each cluster.
        This parameter is ignored if ``misclassified`` is not a float.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` where the selected entries are assigned a label that
        differs from their original one.
    """
    random.seed(seed)

    N = len(X)
    idx = list(range(N))
    classes = sorted(np.unique(y).tolist())
    y_wrong = np.copy(y)

     # ---------- Full random misclassification ---------------
    if isinstance(misclassified, float) and global_misclassification:

        # Number of samples to misclassify in this cluster
        N_misclassified = int(N * misclassified)

        # Get random indices to misclassify
        idx_misclassified = random.sample(idx, N_misclassified)

        y_wrong = np.copy(y)
        for i in idx_misclassified:
            # Get a wrong label (anything but not the correct one, at random)
            new_label = random.sample(list(set(classes) - {y[i]}), 1)[0]

            y_wrong[i] = new_label

    # --------- Random misclassification within clusters ---------------
    else:

        misclassified = set_misclassified(
            misclassified=misclassified, y=y,
            apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
        )

        for c, misclassified_c in zip(classes, misclassified):

            # Indices of the current cluster
            idx_c = [i for i in idx if y[i] == c]
            N_c = len(idx_c)
            # Number of samples to misclassify in this cluster
            N_misclassified = int(N_c * misclassified_c)

            # ------- Random misclassification within cluster -------
            # Get random indices to misclassify
            idx_misclassified = random.sample(idx_c, N_misclassified)

            for i in idx_misclassified:
                # Get new label (anything but not the correct one, at random)
                new_label = random.sample(list(set(classes) - {y[i]}), 1)[0]
                y_wrong[i] = new_label

    return y_wrong


def balanced(
    X: DataArray,
    y: LabelArray,
    misclassified: float = 0.10,
    seed: int = 42,
    allow_same_closest: bool = False,
    apply_misclassification_to_all_clusters: bool = True,
) -> LabelArray:
    """
    Misclassify each cluster evenly toward its nearest centroid.

    The requested number of misclassified samples is distributed evenly
    across clusters. For each cluster, the samples closest to the
    nearest other cluster centroid are relabeled to that neighboring
    cluster.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    allow_same_closest : bool, default=False
        Whether multiple source clusters may target the same closest cluster.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` containing the balanced nearest-centroid relabeling.

    Raises
    ------
    ValueError
        If a cluster does not contain enough samples to satisfy the requested
        per-cluster misclassification count and ``error_if_not_enough`` is
        ``True``.
    """

    random.seed(seed)

    N = len(X)
    idx = list(range(N))
    classes = sorted(np.unique(y).tolist())

    misclassified = set_misclassified(
        misclassified=misclassified, y=y,
        apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
    )

    y_wrong = np.copy(y)

    visited_clusters: list[int] = []
    visited_closest: list[int] = []

    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    for c, misclassified_c in zip(classes, misclassified):

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)
        N_misclassified_c = int(N_c * misclassified_c)

        # ------ Find the closest cluster to c -----------
        distances = np.linalg.norm(centroids - centroids[c], axis=1)

        # Ignore the distance to itself
        distances[c] = np.inf
        if not allow_same_closest:
            # Ignore already visited clusters
            for visited_c in visited_closest:
                distances[visited_c] = np.inf
        closest_cluster = np.argmin(distances)

        # ------- Find closest points to closest cluster ------
        # Get the distances of datapoints in c to this closest cluster
        distances_to_closest = np.linalg.norm(
            X[idx_c] - centroids[closest_cluster], axis=1
        )

        # sort the indices in idx_c
        idx_c_sorted = np.argsort(distances_to_closest)

        idx_misclassified_c = [
            idx_c[i]
            for i in idx_c_sorted[:min(N_misclassified_c, N_c)]
        ]

        # ------- Assign the wrong labels --------------
        for i in idx_misclassified_c:
            y_wrong[i] = closest_cluster

        # ---------- Update visited -------------
        visited_clusters.append(c)
        visited_closest.append(closest_cluster)


    return y_wrong

def bully(
    X: DataArray,
    y: LabelArray,
    misclassified: float = 0.10,
    error_if_not_enough: bool = True,
) -> LabelArray:
    """
    Misclassify datapoints by letting some clusters being absorbed.

    Clusters are processed from the furthest centroid norm and then to
    its neighbors. Samples from a processed cluster are progressively
    reassigned to another cluster until the requested global number of
    misclassified points is met.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : float, default=0.10
        Fraction of samples to relabel across the whole dataset.
    error_if_not_enough : bool, default=True
        Whether to raise a ValueError if there are not enough points in a cluster to satisfy the requested misclassification count.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` after applying the bully-style relabeling.

    Raises
    ------
    ValueError
        If the total number of datapoints minus the number of datapoints
        in the last bully cluster is smaller than the number of samples
        needed to satisfy the requested misclassification count and
        ``error_if_not_enough`` is ``True``.
    """

    N = len(X)
    idx = list(range(N))
    N_misclassified = int(N*misclassified)

    y_wrong = np.copy(y)


    # Compute centroids for each cluster
    centroids = compute_centroids(X, y)

    # Make the furthest away clusters be the first bullied cluster
    # (i.e the one with the greatest norm)
    first_bullied = np.argmax(np.linalg.norm(centroids, axis=1))

    # ------ Find the closest cluster to the first bullied -----------
    distances_to_first = np.linalg.norm(
        centroids - centroids[first_bullied], axis=1
    )

    # Sort the clusters by distance to the first bullied cluster
    sorted_class = np.argsort(distances_to_first)

    # Keep track of indices to misclassify
    idx_bullied: list[int] = []
    N_bullied = 0

    # Bully the first cluster as much as possible, then the second, etc.
    for i, c in enumerate(sorted_class):

        # --------- indices of the current cluster -------
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)

        # Current bully: next cluster in the sorted list
        # If we reached the last bully and still want more misclassified
        # points than N- N_bully, then i+1 will yield an error. This
        # case means that we only have one cluster in the dataset, and
        # we might raise an error if error_if_not_enough is True
        i_bully = i+1
        if i+1 == len(sorted_class):
            if error_if_not_enough:
                raise ValueError(
                    f"Not enough points to misclassify {N_misclassified} points. "
                    f"Only {N - N_bullied} available."
                )
            else:
                bully = c
                idx_bullied += idx_c
        # Regular case, where we have a next cluster to bully
        else:
            bully = sorted_class[i_bully]

            # ------ Find datapoints to bully  -----------
            # Bully as many points as possible from this cluster,
            # but not more than the number of points left to misclassify
            n_newly_bullied = min(N_c, N_misclassified - N_bullied)

            # Find the indices of the closest datapoints to the bully
            if n_newly_bullied < N_c:
                dist_to_bully = np.linalg.norm(
                    X[idx_c] - centroids[bully], axis=1
                )
                # Find in idx_c the indices of the closest points to the bully
                idx_closest_to_bully = np.argsort(dist_to_bully)
                # Find them in idx and keep only the first ones
                idx_closest_to_bully = [
                    idx_c[i] for i in idx_closest_to_bully[:n_newly_bullied]
                ]
            # Otherwise, take them all
            else:
                idx_closest_to_bully = idx_c.copy()

            # Update the list of bullied indices
            idx_bullied += idx_closest_to_bully

        # Update labels if we are done
        N_bullied = len(idx_bullied)

        # Check whether we have enough points to misclassify
        finish = N_bullied >= N_misclassified
        if finish:
            for i in idx_bullied:
                y_wrong[i] = bully

            # Stop
            break

    return y_wrong

def subclustering(
    X: DataArray,
    y: LabelArray,
    misclassified: Union[float, List[float]] = 0.10,
    method: str = "grouped",
    seed: int = 42,
    misclassify_minority: bool = False,
    apply_misclassification_to_all_clusters: bool = True
) -> LabelArray:
    """
    Misclassify datapoints by creating subclusters within some clusters.

    Can create random subclusters or grouped subclusters that are close
    to a random point within the cluster.

    Note that this function a priori keeps the clusters aligned, meaning
    that the labels in ``y`` and ``y_wrong`` will be the same for the
    correctly classified samples.

    If ``misclassify_minority`` is set to ``True``, then the minority of
    the subcluster will be misclassified instead of the majority. More
    specifically, if a subcluster within a cluster becomes the majority
    then the new label of the subcluster will be the same as the
    original cluster label and the minority will be the misclassified
    samples.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : Union[float, List[float]]
        Fraction of samples within a cluster (or a list of clusters if a
        list is provided) whose labels should be replaced. If a list is provided, the length of the list must be less than or equal to the number of clusters in ``y``. If fewer values are provided than the number of clusters, the remaining clusters will have a misclassification rate of 0.
    method : str, default="grouped"
        Method to use for creating subclusters. Can be either "grouped" or "random". If "grouped", the misclassified samples will be grouped together in a subcluster that is close to a random point within the cluster. If "random", the misclassified samples will be randomly selected within the cluster.
    seed : int, default=42
        Seed passed to Python's random generator for reproducibility.
    misclassify_minority : bool, default=False
        Whether to misclassify the minority of the subcluster instead of the majority.
    apply_misclassification_to_all_clusters : bool, default=True
        If ``True`` and if ``misclassified`` is a float, then applies
        the same misclassification rate to all clusters. If ``False``,
        only applies the misclassification strategy to the first
        (smallest label) cluster.

    Returns
    -------
    NDArray[np.int_]
        Copy of ``y`` after applying the relabeling.
    """
    random.seed(seed)

    misclassified = set_misclassified(
        misclassified=misclassified, y=y,
        apply_misclassification_to_all_clusters=apply_misclassification_to_all_clusters
    )

    N = len(X)
    idx = list(range(N))
    # We use set to be able to perform set operations later on
    classes = sorted(np.unique(y).tolist())
    last_label = int(max(classes))
    y_wrong = np.copy(y)

    # Group the misclassified points together in a subcluster
    for c, misclassified_c in zip(classes, misclassified):

        # Ignore this step if no misclassification is requested for this cluster
        if misclassified_c == 0:
            continue

        # Indices of the current cluster
        idx_c = [i for i in idx if y[i] == c]
        N_c = len(idx_c)
        # Number of samples to misclassify in this cluster
        N_misclassified_c = int(N_c * misclassified_c)

        if method == "grouped":

            # Get a random point within the cluster to be the center of the subcluster
            center_idx = random.choice(idx_c)
            center_point = X[center_idx]

            # Compute distances to the center point
            distances_to_center = np.linalg.norm(X[idx_c] - center_point, axis=1)

            # Sort indices by distance to the center point
            idx_c_sorted = np.argsort(distances_to_center)

            # Select the closest points to form the subcluster
            idx_misclassified = [
                idx_c[i]
                for i in idx_c_sorted[:min(N_misclassified_c, N_c)]
            ]
        elif method == "random":
            # Randomly select indices to misclassify
            idx_misclassified = random.sample(idx_c, min(N_misclassified_c, N_c))

        # Case where the subcluster becomes the majority
        if misclassify_minority and len(idx_misclassified) > N_c / 2:
            idx_misclassified = [i for i in idx_c if i not in idx_misclassified]

        last_label = last_label + 1
        for i in idx_misclassified:
            y_wrong[i] = last_label

    return y_wrong

def _label_greatest_cluster(
    classes: List[int],
    N_c: List[int],
) -> int:
    """
    Select the label of the largest cluster among candidate labels.

    Given a list of cluster labels and a corresponding list containing
    cluster sizes, return the label of the greatest number of
    datapoints.

    Parameters
    ----------
    classes : List[int]
        Candidate cluster labels among which a representative label must
        be selected.
    N_c : List[int]
        Cluster-size list indexed by label, where ``N_c[c]`` is the
        number of datapoints in cluster ``c``.

    Returns
    -------
    int
        Label of the largest cluster in ``classes``.

    Notes
    -----
    This helper assumes that all labels in ``classes`` are valid indices
    for ``N_c``.
    """

    # Find which label to keep based on the number of datapoints
    # in each cluster
    kept_label = None
    for c in classes:
        if kept_label is None or N_c[c] >= N_c[kept_label]:
            kept_label = c
    return kept_label

def superclustering(
    X: DataArray,
    y: LabelArray,
    misclassified: float = 0.10,
    superclusters: List[List[int]] = [],
    method: str = "agglomerative",
    method_kwargs = {},
    is_upper_bound: bool = True
) -> LabelArray:
    """
    Build coarser labelings by merging existing clusters.

    This function starts from the original clusters in ``y`` and
    produces a new label array where some clusters are merged together.
    The global amount of relabeling is controlled with ``misclassified``
    and ``is_upper_bound``.

    There are three supported superclustering modes:

    1) Explicit mapping (``superclusters`` argument) - If
       ``superclusters`` is provided (non-empty), it takes priority over
         ``method``.
       - Each list ``([clusters_to_merge])`` forces all listed clusters
         to receive the label with the greatest number of datapoints.
       - This is a manual, fully user-defined merging strategy.

    2) Size-based merging (``method="smallest"``) - Repeatedly merges
       the two current smallest clusters/superclusters. - After each
       merge, the merged supercluster size is updated and used
         for the next ordering step.
       - Distances are ignored on purpose; this yields intentionally
         naive superclustering examples.

    3) Hierarchical centroid merging (``method="agglomerative"``) -
       Computes one centroid per original cluster. - Runs
       ``sklearn.cluster.AgglomerativeClustering`` on those centroids
         to obtain a merge tree.
       - Replays tree merges until the requested misclassification
         budget is reached (or just before crossing it when
         ``is_upper_bound=True``).
       - ``method_kwargs`` are forwarded to ``AgglomerativeClustering``.

    Misclassification accounting
    ----------------------------
    Let ``N_to_misclassify = int(len(X) * misclassified)``. - If
    ``is_upper_bound=True``, the function stops before a merge that
      would exceed ``N_to_misclassify``.
    - If ``is_upper_bound=False``, the function keeps merging until the
      budget is reached or crossed.

    Parameters
    ----------
    X : NDArray[np.float64]
        Dataset of shape ``(n_samples, n_features)``.
    y : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    misclassified : float, default=0.10
        Target global fraction of samples that should become relabeled
        after cluster merges.
    superclusters : List[List[int]], default=[]
        Optional explicit merge specification. If non-empty, overrides
        ``method``.
    method : str, default="agglomerative"
        Automatic merge strategy when ``mapping`` is empty. Supported
        values are ``"smallest"`` and ``"agglomerative"``.
    method_kwargs : dict, default={}
        Extra keyword arguments forwarded to
        ``sklearn.cluster.AgglomerativeClustering`` when
        ``method="agglomerative"``.
    is_upper_bound : bool, default=True
        Whether to treat ``misclassified`` as an upper bound (strict
        stop before crossing) or as a lower bound (allow final
        crossing).

    Returns
    -------
    LabelArray
        Copy of ``y`` after superclustering merges.

    Raises
    ------
    NotImplementedError
        If ``method`` is not supported.
    """
    y_wrong = np.copy(y)
    classes = sorted(np.unique(y).tolist())

    # Number of datapoints in original clusters
    N_c = {c: int(np.sum(y == c)) for c in classes}

    # Threshold in terms of number of datapoints to misclassify
    N_to_misclassify = int(len(X) * misclassified)

    # Keep a running count of how many samples have been relabeled.
    N_misclassified = 0

    if len(classes) <= 1:
        return y_wrong
    elif superclusters:
        # Explicit mapping
        for classes_to_merge in superclusters:

            # Find which label to keep based on the number of datapoints
            kept_label = _label_greatest_cluster(classes_to_merge, N_c)

            for c in classes_to_merge:
                y_wrong[y == c] = kept_label

    elif method == "smallest":
        # Keep merging together the smallest clusters as long as we have
        # not reached the final number of misclassified points.
        #
        # After merging 2 clusters together, it is their new number of
        # datapoints that count in the ordering of the next clusters to
        # merge together Note that this method doesn't take into account
        # the distances between the clusters, but this is on purpose, to
        # get examples of superclustering that are not good
        # superclustering.

        # components: each component (index in list) represent a (super)class
        components: list[set[int]] = [{c} for c in classes]
        # component_sizes: actual size of the (super)classes
        component_sizes: list[int] = [N_c[c] for c in classes]
        # sorted_comp: argsort of the components based on their size
        sorted_comp: list[int] = []
        # merged_classes: tracks which original classes have already
        # been absorbed into a supercluster
        merged_classes: set[int] = set()

        while len(components) > 1:

            # Update the order of the components based on their current sizes
            sorted_comp = sorted(
                range(len(component_sizes)), key=lambda i: component_sizes[i]
            )
            # Pick the two smallest current components.
            i_smallest, j_smallest = sorted_comp[0], sorted_comp[1]
            # Get the classes represented by the components to merge
            classes_to_merge = components[i_smallest] | components[j_smallest]

            # Find which label to keep based on the number of datapoints
            kept_label = _label_greatest_cluster(classes_to_merge, N_c)

            # newly_misclassified: number of points that are not
            # already among the merged classes and that are not in the
            # kept label.
            newly_misclassified = sum(
                N_c[c] for c in classes_to_merge
                if c not in merged_classes and c != kept_label
            )

            # Check whether we should stop before completing this step.
            if (
                is_upper_bound
                and N_misclassified + newly_misclassified > N_to_misclassify
            ):
                break

            # Apply the new label to all merged clusters
            for c in classes_to_merge:
                y_wrong[y == c] = kept_label

            # Update counters
            merged_classes.update(classes_to_merge)
            N_misclassified += newly_misclassified

            # ----------- Update components ---------------
            merged_size = component_sizes[i_smallest] + component_sizes[j_smallest]
            # Remove components that were merged
            # Note that we need to remove the largest index first as to
            # not perturb the index of the second one
            for idx_to_remove in sorted((i_smallest, j_smallest), reverse=True):
                del components[idx_to_remove]
                del component_sizes[idx_to_remove]

            # Add the resulting super component
            components.append(classes_to_merge)
            component_sizes.append(merged_size)

            # In lower-bound mode we allow the final merge to cross the
            # target, but we still stop as soon as the budget is reached.
            if N_misclassified >= N_to_misclassify and not is_upper_bound:
                break

    elif method == "agglomerative":

        # Build the hierarchy on the centroids of the original clusters.
        #
        # Although not optimal, this facilitate things as it allow us to
        # start from datapoints instead of starting the agglomerative
        # clustering from some pre-defined clusters
        centroids = compute_centroids(X, y)

        # Build the full agglomerative tree over the centroids.
        # ``distance_threshold=0`` and ``n_clusters=None`` force sklearn
        # to keep merging until a single tree is built, so we can stop
        # later exactly when our misclassification budget is reached.
        agglomerative = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_distances=True,
            **method_kwargs,
        )
        agglomerative.fit(centroids)

        # id_to_class: maps a tree node id to the set of original
        # cluster labels it contains.
        id_to_class: dict[int, set[int]] = {c: {c} for c in classes}

        # merged_classes: tracks which original classes have already
        # been absorbed into a supercluster
        merged_classes: set[int] = set()

        for merge_index, (left, right) in enumerate(agglomerative.children_):
            left = int(left)
            right = int(right)

            # Retrieve the original clusters contained in the two child
            # nodes being merged by the agglomerative tree.
            classes_to_merge = id_to_class[left] | id_to_class[right]

            # Find which label to keep based on the number of datapoints
            kept_label = _label_greatest_cluster(classes_to_merge, N_c)

            # newly_misclassified: number of points that are not
            # already among the merged classes and that are not in the
            # kept label.
            newly_misclassified = sum(
                N_c[c] for c in classes_to_merge
                if c not in merged_classes and c != kept_label
            )

            # Check whether we should stop before completing this step.
            if (
                is_upper_bound
                and N_misclassified + newly_misclassified > N_to_misclassify
            ):
                break

            # Apply the new label to all merged clusters
            for c in classes_to_merge:
                y_wrong[y == c] = kept_label

            # Update counters
            merged_classes.update(classes_to_merge)
            N_misclassified += newly_misclassified

            # ------------------ Update tree map ----------------------
            # Register the newly created tree node so it can participate in
            # later merges. sklearn uses node ids after the original leaves.
            id_to_class[len(classes) + merge_index] = classes_to_merge

            # In lower-bound mode we allow the final merge to cross the
            # target, but we still stop as soon as the budget is reached.
            if N_misclassified >= N_to_misclassify and not is_upper_bound:
                break

    else:
        raise NotImplementedError(f"Unsupported superclustering method: {method!r}")
    return y_wrong


def flag_misclassified(
        y_true: LabelArray,
        y_wrong: LabelArray
    ) -> LabelArray:
    """
    Flag the misclassified samples between two label arrays.

    Keeps all labels from the true array as is except for datapoints that were misclassified in y_wrong for which a new label `-1` is added to the returned array.,

    Note that this function assumes that the labels in ``y_true`` and ``y_wrong`` are "aligned".

    Note that ``y_true`` and ``y_wrong`` can have different number of clusters, but the labels must be aligned for the correctly classified samples.

    Parameters
    ----------
    y_true : NDArray[np.int_]
        Ground-truth cluster labels encoded as integers.
    y_wrong : NDArray[np.int_]
        Predicted cluster labels encoded as integers.

    Returns
    -------
    LabelArray
        Copy of ``y_true`` except that the misclassified entries are assigned a label `-1`.
    """
    y_flagged = np.copy(y_true)
    y_flagged[y_true != y_wrong] = -1
    return y_flagged
