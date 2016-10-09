"""
Just playing around with clustering. This file is "divisive hierarchical clustering".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Creating some data to work with...")

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10

df = pd.DataFrame(X, columns=variables, index=labels)

print(df)


print("Calculating the distance matrix for the given data...")

from scipy.spatial.distance import pdist, squareform

row_dist = pd.DataFrame(
                        squareform(pdist(df, metric="euclidean")),
                        columns=labels,
                        index=labels
                        )

print(row_dist)


print("Creating clusters based on the data...")

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(pdist(df, metric="euclidean"), method="complete")

data_frame_view = pd.DataFrame(
                                row_clusters,
                                columns=["row label 1", "row label 2", "distance", "no. of items in cluster"],
                                index=["cluster %d" % (i + 1) for i in range(row_clusters.shape[0])]
                                )

print(data_frame_view)


print("Creating tree diagram of clusters...")

from scipy.cluster.hierarchy import dendrogram

row_dendr = dendrogram(row_clusters, labels=labels)
plt.tight_layout()
plt.ylabel("Euclidean distance")
plt.show()














