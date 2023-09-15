# ------ Select the most relevant clustering algorithm ------

# For each dataset, find all experiments working on this dataset but using different clustering methods (by filtering on the filename)
# Compute VI between the true clustering and each clustering obtained with the different clustering method with the real number of clusters
# Keep only the clustering method with the lowest VI and copy the experiment file and the figures to another folder