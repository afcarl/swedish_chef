"""
Module to hold a RecipeTable class.
"""

import pickle


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

    def get_recipes(self):
        """
        Gets all the recipes from the table.
        @return: The list of RecipeObjects.
        """
        return self.__recipes


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
