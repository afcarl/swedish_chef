"""
Module to hold the Cluster class.
"""

import os
from tqdm import tqdm
import warnings
import myio.myio as myio
import chef_global.debug as debug
import chef_global.config as config


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
    cluster_paths = [os.path.join(config.CLUSTERS, f) for f in os.listdir(config.CLUSTERS)\
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

        print("    |-> Total number of clusters to regenerate: " + str(len(clusters)))
        print("    |-> Started at: " + myio.print_time())
        for recipe in tqdm(rec_table):
            for ingredient in recipe:
                ingredients.append(ingredient)
        print("Number of ingredients: " + str(len(ingredients)))
        print("Removing duplicates from ingredients...")
        ingredients = set(ingredients)
        print("Number of unique ingredients: " + str(len(ingredients)))
        print("Pairing each ingredient with its predicted index...")
        tups = []
        for ingredient in tqdm(ingredients):
            fv = rec_table.ingredient_to_feature_vector(ingredient)
            predicted_index = (kmeans.predict(fv))[0]
            as_tup = (ingredient, predicted_index)
            tups.append(as_tup)
        print("Sorting ingredients...")
        sorted(tups, key=lambda t: t[1])
        print("Looping over sorted ingredients and clusters...")
        tups = list(tups)
        last_index = (tups[0])[1]
        print("last index: " + str(last_index))
        for tup in tqdm(tups):
            ingredient, this_index = tup
            if this_index != last_index:
                print("Done with cluster index: " + str(last_index))
                cluster = clusters[last_index]
                myio.save_pickle(cluster, config.CLUSTERS + "cluster" + str(cluster.get_index()))
            else:
                print("This index: " + str(this_index))
                cluster = clusters[this_index]
                cluster.add(ingredient)

        # Untenably slow
        #already_done = {}
        #for cluster in clusters:
        #    print("Working on cluster " + str(cluster.get_index()) + " of 499...")
        #    for recipe in tqdm(rec_table):
        #        for ingredient in recipe:
        #            if ingredient not in already_done:
        #                fv = rec_table.ingredient_to_feature_vector(ingredient)
        #                predicted_index = (kmeans.predict(fv))[0]
        #                if (cluster.get_index() == predicted_index):
        #                    cluster.add(ingredient)
        #                    already_done[ingredient] = True
        #    myio.save_pickle(cluster, config.CLUSTERS + "cluster" + str(cluster.get_index()))

        #print("    |-> Started at: " + myio.print_time())
        #for recipe in tqdm(rec_table):
        #    for ingredient in recipe:
        #        fv = rec_table.ingredient_to_feature_vector(ingredient)
        #        predicted_index = (kmeans.predict(fv))[0]
        #        clusters[predicted_index].add(ingredient)
        #        debug.debug_print("Adding " + str(ingredient) + " to " + str(predicted_index))
        #print("    |-> Finished at: " + myio.print_time())

        #for cluster in clusters:
        #    myio.save_pickle(cluster, config.CLUSTERS + "cluster" + str(cluster.get_index()))









