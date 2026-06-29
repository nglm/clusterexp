"""
General functions on datasets

This module assumes that the data is already formatted as expected, with a file for the data and a file for the labels. For functions specific to Barton's datasets or the UCR dataset, see barton.py or ucr.py.

Functions defined here are general to all datasets.
"""

def print_heads(
    fnames: List[str],
    path:str = "./",
    n_labels_max:int = 20,
    n_samples_max:int = 10000,
    UCR: bool = False,
) -> None:
    """
    Print summary information and heads for multiple datasets.

    Parameters
    ----------
    fnames : List[str]
        Dataset names or filenames.
    path : str, optional
        Prefix used to resolve each dataset location, by default "./".
    n_labels_max : int, optional
        Threshold used to flag datasets with too many labels,
        by default 20.
    n_samples_max : int, optional
        Threshold used to flag datasets with too many samples,
        by default 10000.
    UCR : bool, optional
        If True, load UCR-formatted files from local TSV paths;
        otherwise load ARFF datasets, by default False.

    Returns
    -------
    Dict[str, Dict]
        Per-dataset summary containing metadata such as shape,
        labeling information, and potential loading errors.
    """
    print(f"MAX LABELS: {n_labels_max}\nMAX SAMPLES: {n_samples_max}\n")
    summary = {}
    for f in fnames:
        summary[f] = {}

        # Get the dataframe corresponding to the filename
        # We don't use get_data_labels functions here because we want to
        # use the raw df.
        if UCR:
            fname = get_fname(f, only_root=False, data_source='UCR')
            print(fname)
            full_f = path+fname
            try:
                df = pd.read_csv(full_f, sep="\t")
            except Exception as ex:
                meta = ex
                df = None
        else:
            full_f = path + f
            print(full_f)
            data, meta = arff_from_github(full_f)
            if data is None:
                df = None
            else:
                df = pd.DataFrame(data)
        # Print the head of the data frame, to get a better idea of the
        # dataset

        if df is not None:
            cols = df.columns.str.lower()

            labeled = (("class" in cols) or UCR)
            has_na = df.isnull().sum().sum() > 0
            shape = (len(df), len(cols))


            # We use get_data_labels here just to count the labels,
            # not to get df as it would already be processed
            if UCR:
                _, _, n_labels, _ = get_data_labels_UCR(full_f, path="")
            else:
                if "class" in cols:
                    _, _, n_labels, _ = get_data_labels(full_f, path="")
                else:
                    n_labels = None

            if labeled:
                too_many_labels = n_labels > n_labels_max
            else:
                too_many_labels = False

            msg = (
                f"Shape: {shape}   |   n_labels: {n_labels}\n" +
                f"Labeled:         {labeled}\n" +
                f"Has NA values:   {has_na}\n" +
                f"Too many labels: {too_many_labels}\n" +
                f"Too many samples:{shape[0]>n_samples_max}"
            )
            print(msg)
            print(df.head())

            summary[f]["labeled"] = labeled
            summary[f]["has_na"] = has_na
            summary[f]["shape"] = shape
        # If there was a problem loading the data, then the error message
        # is returned in "meta"
        else:
            summary[f]["error"] = meta
            print(meta)
    return summary


def process_labels(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Encode labels and infer the effective count.

    Parameters
    ----------
    labels : np.ndarray
        Original label values.

    Returns
    -------
    Tuple[np.ndarray, int]
        Encoded labels and the number of effective classes. If each
        sample has a unique label, labels are collapsed to one class.
    """
    N = len(labels)
    classes = np.unique(labels)
    map_classes = {c:i for i,c in enumerate(classes)}
    n_labels = len(classes)
    if n_labels == N:
        n_labels = 1
        labels = np.zeros_like(labels, dtype=int)
    else:
        labels = np.array(
            [map_classes[label] for label in labels],
            dtype=int)
    return labels, n_labels


def get_list_datasets(fname: str) -> List[str]:
    """
    Read a file containing one dataset name per line.

    Parameters
    ----------
    fname : str
        Path to the file with dataset names.

    Returns
    -------
    List[str]
        Dataset names.
    """
    with open(fname) as f:
        datasets = f.read().splitlines()
    return datasets


def write_list_datasets(fname:str, lines: List[str]) -> None:
    """
    Write dataset names to a text file, one per line.

    Parameters
    ----------
    fname : str
        Output file path.
    lines : List[str]
        Dataset names to write.
    """
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))