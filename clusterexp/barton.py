URL_ROOT = 'https://raw.githubusercontent.com/nglm/clustering-benchmark/master/src/main/resources/datasets/'

N_SAMPLES_MAX = 10000

# Just one cluster
UNIMODAL = [
    "birch-rg1.arff", "birch-rg2.arff",
    "golfball.arff",
]
# No class column in the data
UNLABELED = [
    # artificial
    "birch-rg1.arff", "birch-rg2.arff",
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",

    # real-world
    "water-treatment.arff",
]

# Unknown number of clusters
UNKNOWN_K = [
    # artificial
    "birch-rg3.arff",
    "mopsi-finland.arff", "mopsi-joensuu.arff",
    "s-set3.arff", "s-set4.arff",

    # real-world
]

# Datasets removed, for various reasons (e.g. missing data)
INVALID = [
    #"segment.arff",
    # Contains missing values
    'dermatology.arff',
    "water-treatment.arff",
    # give arff error: "String attributes not supported yet, sorry"
    "yeast.arff",
]



# Too many labels
# (More than 20 in non-time series data, more than 15 in UCR)
TOO_MANY_LABELS = [
    # artificial
    "D31.arff", "fourty.arff",
    # real-world
    "cpu.arff", "letter.arff",
]

# Too many samples
# (More than 10000)
TOO_MANY_SAMPLES = [
    # artificial
    "mopsi-finland.arff", "birch-rg3.arff", "birch-rg2.arff",
    "birch-rg1.arff",
    # real-world
    "letter.arff",
]

def arff_from_github(url, verbose=False):
    """
    Load an ARFF dataset from a URL.

    Parameters
    ----------
    url : str
        URL pointing to an ARFF file.
    verbose : bool, optional
        If True, print the HTTP status code, by default False.

    Returns
    -------
    Tuple[Union[None, np.ndarray], Union[Exception, arff.MetaData]]
        Parsed ARFF data and metadata when successful. If loading fails,
        returns ``(None, exception)``.
    """
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if verbose:
                print(response.status, flush=True)
            arff_data = io.StringIO(response.read().decode('utf-8'))
            data, meta = arff.loadarff(arff_data)
    except Exception as ex:
        print(ex, flush=True)
        return None, ex
    return data, meta

def load_data_from_github(
    url: str,
    with_labels: bool = True
) -> Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]:
    """
    Return data, labels, and metadata from an GitHub ARFF URL.

    Non-numerical variables are ignored.

    Parameters
    ----------
    url : str
        URL of the dataset.
    with_labels : bool, optional
        If True, include labels from the ``class`` column, by default
        True.

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], arff.MetaData]
        Numeric data array, optional labels array, and ARFF metadata.
    """
    data, meta = arff_from_github(url)
    df = pd.DataFrame(data)
    df.columns = df.columns.str.lower()
    # We keep only numerical variables
    data_col = [
        c for c, t in zip(df.columns, df.dtypes)
        if (c != "class") and t in ["float", "int"]
    ]
    # Get only data, not the labels and convert to numpy
    if with_labels:

        data = df[data_col].to_numpy()
        labels = df["class"].to_numpy()
    else:
        data = df[data_col].to_numpy()
        labels = None
    return data, labels, meta

def get_data_labels(
    fname: str,
    path: str ="./"
) -> Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]:
    """
    Get dataset, labels, number of labels, and metadata for non UCR data

    Parameters
    ----------
    fname : str
        Dataset filename.
    path : str, optional
        Prefix path or URL for the dataset, by default "./".

    Returns
    -------
    Tuple[np.ndarray, Union[None, np.ndarray], int, arff.MetaData]
        Data array, optional labels, inferred number of labels, and
        ARFF metadata.
    """
    n_labels = None
    if fname in UNLABELED:
        with_labels = False
        if fname in UNIMODAL:
            n_labels = 1
        else:
            n_labels = None
    else:
        with_labels = True
    data, labels, meta = load_data_from_github(
        path + fname, with_labels=with_labels
    )
    if with_labels:
        labels, n_labels = process_labels(labels)
    return data, labels, n_labels, meta