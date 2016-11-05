"""
Module to hold a RecipeTable class.
"""

import math
import warnings
import random
import pickle
import scipy.sparse as sparse
from tqdm import tqdm
import myio.myio as myio
import statistics.cluster as cluster
import chef_global.debug as debug


class RecipeTable:
    """
    A class to hold all of the recipes.
    Can be saved and loaded to/from disk.
    """
    def __init__(self, recipes=[]):
        self.__recipes = []
        for recipe in recipes:
            self.__recipes.append(recipe)
        self.__clusters = []

    def __iter__(self):
        for recipe in self.__recipes:
            yield recipe

    def __len__(self):
        return len(self.__recipes)

    def add_recipe(self, recipe):
        """
        Takes a Recipe object and adds it to
        the list of recipes held here.
        @param recipe: The recipe to add.
        @return: void
        """
        self.__recipes.append(recipe)

    def calculate_typical_consecutive_zeros(self):
        """
        Calculates the typical number of consecutive zeros
        in a typical feature vector. Useful for Golomb compression.
        @return: The typical number of consecutive zeros
        """
        return 1268.1495099738602 # Cached value (since calculating it takes 5 hours)

        print("Calculating the typical number of zeros...")
        num_ingredients = 0
        total_total = 0
        for recipe in tqdm(self):
            for ingredient in recipe:
                num_ingredients += 1
                fv = self.ingredient_to_feature_vector(ingredient)
                last = 0
                num_sequences = 0
                num_consec = 0
                total = 0
                for r in fv:
                    if r is 1 and last is 0:
                        num_sequences += 1
                        total += num_consec
                        num_consec = 0
                        last = 1
                    elif r is 1 and last is 1:
                        last = 1
                    elif r is 0 and last is 1:
                        num_consec += 1
                        last = 0
                    elif r is 0 and last is 0:
                        num_consec += 1
                        last = 0
                avg_consec = total / num_sequences
                total_total += avg_consec
        total_avg = total_total / num_ingredients
        return total_avg

    def compute_stats(self):
        """
        Computes the average and standard deviation of
        recipe lengths.
        @return: tuple of the form (avg, stdev)
        """
        s = 0
        for rec in self:
            s += len(rec)
        avg = s / len(self)

        s = 0
        for rec in self:
            l = len(rec)
            diff = l - avg
            diff_sqr = diff * diff
            s += diff_sqr
        std = math.sqrt(s / len(self))

        return avg, std

    def get_all_ingredients(self):
        """
        Gets the set of all unique ingredients found in all
        of the recipes.
        @return: The ingredients
        """
        ingredients = []
        for recipe in self:
            for ingredient in recipe:
                ingredients.append(ingredient)
        return set(ingredients)

    def get_cluster(self, index):
        """
        Gets the cluster with the given index.
        @param index: The index to use to retrieve the cluster
        @return: A cluster object whose index is index
        """
        if index >= len(self.__clusters):
            return None
        else:
            return self.__clusters[index]

    def get_random_ingredient(self, seed=0):
        """
        Gets a random ingredient from the table.
        @param seed: The random seed.
        @return: random ingredient
        """
        random.seed(seed)
        recipe = []
        while len(recipe) == 0:
            recipe_index = random.randint(0, len(self.__recipes) - 1)
            recipe = self.__recipes[recipe_index]

        if len(recipe) == 0:
            return recipe[0]
        else:
            ingredient_index = random.randint(0, len(recipe) - 1)
            ingredient = recipe[ingredient_index]
            return ingredient

    def get_random_number(self):
        """
        Gets a random number of ingredients from a Gaussian distribution
        centered around the average number of ingredients in a recipe.
        Can return 0, but no negatives.
        @return: The random number.
        """
        avg, std = self.compute_stats()
        num = random.gauss(avg, std)
        num = 0 if num < 0 else num
        return int(num + 0.5)

    def get_recipes(self):
        """
        Gets all the recipes from the table.
        @return: The list of RecipeObjects.
        """
        return self.__recipes

    def ingredient_to_feature_vector(self, ingredient):
        """
        Generates a vector of the form:
        Recipe 0    Recipe 1    Recipe 2    ...
            0          1           0
        Where a 0 means the ingredient is not present in that recipe and a 1
        means it is.
        This means the return value is a list of the form [0, 1, 0, ...] where
        each index represents a recipe and the value at that index represents
        present or not.
        @param ingredient: The ingredient for which to generate a feature vector
        @return: The feature vector
        """
        to_ret = [1 if recipe.has(ingredient) else 0 for recipe in self.__recipes]
        return to_ret

    def ingredient_to_feature_vector_sparse(self, ingredient):
        """
        Does exactly the same thing as ingredient_to_feature_vector, but
        in a sparse format, specifically, it returns a csr_matrix.
        """
        list_form = self.ingredient_to_feature_vector(ingredient)
        lil_matrix_form = sparse.lil_matrix(list_form)
        return lil_matrix_form.tocsr()

    def load_in_clusters(self, clusters):
        """
        Loads the clusters into the recipe table.
        @param clusters: list of cluster objects
        @return: void
        """
        max_index = 0
        for c in clusters:
            max_index = c.index if c.index > max_index else max_index
        for i in range(max_index + 1):
            self.__clusters.append("empty")
        for c in clusters:
            self.__clusters[c.index] = c

def save_to_disk(obj, path="tmp/recipe_table"):
    """
    Saves the object to the disk in the given path.
    @param obj: The object to save.
    @param path: The path to save at.
    @return: The path it was saved to.
    """
    pickle.dump(obj, open(path, 'wb'))
    return path

def load_from_disk(path="tmp/recipe_table"):
    """
    Loads the pickled object from the path.
    @param path: The path to the pickled object.
    @return: The object.
    """
    obj = pickle.load(open(path, 'rb'))
    return obj
