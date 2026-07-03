from pathlib import Path

home_dir = Path.home()


from clusterexp.barton import (
    save_data_labels_from_github, get_list_datasets_from_github
)
from clusterexp.data import write_list_datasets

PATH_DATA = f"{home_dir}/Documents/Syncthing/Data/ClusterExp-restart2026/Barton/"

data_sources = ["artificial", "real-world"]

for data_source in data_sources:

    # ---------------- Lists of datasets from GitHub ------------
    # Getting the lists of datasets from GitHub
    all_datasets = get_list_datasets_from_github(data_source=data_source)
    valid_datasets = get_list_datasets_from_github(
        data_source=data_source, with_invalid=False, with_unknown_k=False
    )

    # Saving the lists of datasets to text files
    write_list_datasets(
        f"{PATH_DATA}{data_source}-datasets-all.txt", all_datasets
    )
    write_list_datasets(
        f"{PATH_DATA}{data_source}-datasets-valid.txt", valid_datasets
    )


    # ---------------- Save locally data and labels ------------
    save_data_labels_from_github(
        dataset_names=valid_datasets,
        path_data=f"{PATH_DATA}{data_source}/",
        data_source=data_source
    )



