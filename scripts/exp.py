import sklearn
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
import numpy
# from numpy.random import RandomState

from clusterexp.utils import write_json, load_json

fname = "test_config.json"
write_json(fname, config)

loaded_config = load_json(fname)

print(type(loaded_config["model_class"]), loaded_config["model_class"])