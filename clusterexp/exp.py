import sys
from datetime import datetime

from .config import load_config_as_dict
from .data import find_datasets, filter_datasets
from .utils import save_log, print_log

def prepare_data(config_fname:str) -> dict:
    """
    - Reads the config file: function `load_config_as_dict`
    - Prints the initiated log file containing the config file used `print_log` function
    - Find all datasets in the path_data folder (using the function `find_datasets`)
    - filter datasets based on the path_data and constraints defined in the config file (using the function `filter_datasets`)
      - Creates a list of kept datasets
      - Creates the dictionnary of sorted (kept, and one key per constraint) datasets
    - Creates a json resulting logfile `log-data-20XX-XX-XX.json`, with
      - One key `config_data` with the corresponding dict of the data config file used
      - One key `log_data` with the corresponding dict:
          - `dropped_datasets` a dict of lists of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `kept_datasets` list of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `log_fname` : `path/to/res/log-data-20XX-XX-XX` (without the `.txt` or `.json`)
    - Save the merged dictionary ``log-data-20XX-XX-XX.json``  with the function `save_log`
    - Save output log file ``log-data-20XX-XX-XX.txt``
    - Returns the merged dictionary ``log-data-20XX-XX-XX.json`` as a dict
    """
    # ---------------- Read config file ---------------------
    config = load_config_as_dict(config_fname)
    path_data = config['config_data']['path_data']
    path_res = config['config_data']['path_res']

    # ---------------- Prepare log files ---------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
    log_fname = f'{path_res}log-data-{full_date}'
    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        "config_data": config['config_data'],
        "log_data": {
            "log_fname": log_fname,
        }}

    print_log(log)

    # ---------------- Find datasets ------------------------
    datasets = find_datasets(path_data)

    # --------------- Filter datasets ----------------------
    constraints = config['config_data'].copy()
    constraints.pop('path_data', None)
    constraints.pop('path_res', None)

    filtered_datasets = filter_datasets(datasets, **constraints)

    log['log_data'].update(filtered_datasets)

    # ---------------- Save log files ------------------------
    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )

    print_log(log)

    fout.close()
    return log