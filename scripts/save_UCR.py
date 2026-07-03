"""
Format UCR dataset to match requirements of ClusterExp

- Take all `_train.tsv` datasets and save them into 2 files: one for the data and one for the labels.
  - The data is saved as a numpy array of shape (N, 1, T)
  - The labels are saved as a numpy array of shape (N,).

- We ignore datasets with missing values and variable length, instead, we take the corresponding datasets from ``Missing_value_and_variable_length_datasets_adjusted``
- a priori ``Missing_value_and_variable_length_datasets_adjusted`` doesn't have to be treated differently, it will just have a longer ``path/to/dataset``.
- Sometimes the train size is much smaller than the test size, we do as if we didn't notice that, the clustering will be bad at that's it, it will just not contribute to the final results. We will mention that though in the paper, but we don't remove manually those datasets.
- We don't have a "too_few_samples" contraints because it depends so much on the number of clusters, the separability between classes and having too few samples is not a burden in terms of computations contrary to having too many samples.

- Save a list of all datasets and a list of valid datasets to text files.
"""

from pathlib import Path

home_dir = Path.home()

PATH_UCR_LOCAL = f"{home_dir}/Documents/Syncthing/Data/UCR/UCRArchive_2018/"


from clusterexp.ucr import (
    save_data_labels_UCR, find_datasets_UCR
)
from clusterexp.data import write_list_datasets

PATH_DATA = f"{home_dir}/Documents/Syncthing/Data/ClusterExp-restart2026/UCR/"

# ---------------- Lists of datasets from Local ------------
# Getting the lists of datasets from GitHub
all_datasets = find_datasets_UCR(
    path_ucr=PATH_UCR_LOCAL, with_ill_formated=True
)
valid_datasets = find_datasets_UCR(
    path_ucr=PATH_UCR_LOCAL, with_ill_formated=False
)

# Saving the lists of datasets to text files
write_list_datasets(
    f"{PATH_DATA}datasets-all.txt", all_datasets
)
write_list_datasets(
    f"{PATH_DATA}datasets-valid.txt", valid_datasets
)


# ---------------- Save locally data and labels ------------
save_data_labels_UCR(
    fnames=valid_datasets,
    path_ucr=PATH_UCR_LOCAL,
    path_data=f"{PATH_DATA}",
)




