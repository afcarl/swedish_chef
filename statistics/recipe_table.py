"""
Module to hold a RecipeTable class.
"""

import random
import pickle
import scipy.sparse as sparse


class RecipeTable:
    """
    A class to hold all of the recipes.
    Can be saved and loaded to/from disk.
    """
    def __init__(self, recipes=[]):
        self.__recipes = []
        for recipe in recipes:
            self.__recipes.append(recipe)

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

        print("Randomly chosen recipe: " + str(recipe))
        if len(recipe) == 0:
            return recipe[0]
        else:
            ingredient_index = random.randint(0, len(recipe) - 1)
            ingredient = recipe[ingredient_index]
            return ingredient

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
