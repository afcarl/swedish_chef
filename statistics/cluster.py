"""
Module to hold the Cluster class.
"""

import os
from tqdm import tqdm
import warnings
import myio.myio as myio
import chef_global.debug as debug


class Cluster:
    """
    Class to hold a kmeans cluster of ingredients.
    """

    def __init__(self, index, ingredients=[]):
        self.index = index
        self.ingredients = []
        for ingredient in ingredients:
            self.ingredients.append(ingredient)

    def add(self, ingredient):
        self.ingredients.append(ingredient)

    def get_index(self):
        return self.index


def load_clusters():
    """
    Loads the clusters from the disk. Returns them as a list
    of clusters.
    @return: List of clusters
    """
    cluster_paths = [f for f in os.listdir(config.CLUSTERS)\
                        if os.path.isfile(os.path.join(config.CLUSTERS, f))]
    clusters = [myio.load_pickle(cluster_path) for cluster_path in cluster_paths]
    return clusters


def regenerate_clusters(kmeans, rec_table):
    """
    Goes through each ingredient in each recipe in the table and
    caches each one into the right cluster according to the passed
    in kmeans. This will take an OUTRAGEOUS amount of time, but
    should only have to be done once and will allow O(1) lookup
    for retrieving whole clusters, which will then allow super
    fast searching of those clusters.
    Then saves them to disk.
    @param kmeans: The trained kmeans cluster model
    @param rec_table: The table
    @return: list of all of the clusters
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        labels = set(kmeans.labels_)
        clusters = []
        for i, label in enumerate(labels):
            clusters.append(Cluster(i))

        print("    |-> Started at: " + myio.print_time())
        for recipe in tqdm(rec_table):
            for ingredient in recipe:
                fv = rec_table.ingredient_to_feature_vector(ingredient)
                predicted_index = (kmeans.predict(fv))[0]
                clusters[predicted_index].add(ingredient)
                debug.debug_print("Adding " + str(ingredient) + " to " + str(predicted_index))
        print("    |-> Finished at: " + myio.print_time())

        for cluster in clusters:
            myio.save_pickle(cluster, config.CLUSTERS + "cluster" + str(cluster.get_index()))








