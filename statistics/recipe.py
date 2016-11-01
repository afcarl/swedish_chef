"""
A module to hold the Recipe class.
"""

import os

class Recipe:
    """
    Class to represent a recipe - a list of ingredients (and potentially steps).
    """

    def __init__(self, table, ingredients=[]):
        """
        Constructor.
        @param table: The IngredientsTable to use as a reference.
        @param ingredients: An optional list of ingredients to initialize the
                            recipe with. Duplicate items in the list are ignored.
                            The ingredients must be found in the
                            table or a KeyError will be thrown.
        """
        self.__table = table
        self.__ingredients_list = []
        for ingredient in ingredients:
            self.__ingredients_list.append(ingredient)
        self.__ingredients_list = list(set(self.__ingredients_list))

    def __getitem__(self, key):
        return self.__ingredients_list[key]

    def __iter__(self):
        for ingredient in self.get_ingredients_list():
            yield ingredient

    def __len__(self):
        return len(self.get_ingredients_list())

    def __str__(self):
        to_ret = "Recipe:" + os.linesep
        for ingredient in self.__ingredients_list:
            i_d = self.__table.get_id(ingredient)
            to_ret += str(i_d) + ":    " + str(ingredient) + os.linesep
        return to_ret

    def get_feature_vector(self):
        """
        Returns this recipe as a feature vector of the form:
        [0, 1, 0, 0, 0, 1, 0, ...], where a given index in the
        vector corresponds to an ID from the IngredientsTable,
        and a 1 at that location indicates that this recipe contains
        that ingredient, while a 0 means that this recipe does not
        contain that ingredient.
        @return: The feature vector
        """
        fv = [0 for i in range(len(self.__table))]
        for ingredient in self.get_ingredients_list():
            i_d = self.__table.get_id(ingredient)
            fv[i_d] = 1
        return fv

    def get_ingredients_ids(self):
        """
        Gets a list of IDs corresponding to all the ingredients that
        are in this recipe.
        @return: A list of ingredients IDs.
        """
        ids = [self.__table.get_id(ingredient) for ingredient in self.__ingredients_list]
        return ids

    def get_ingredients_list(self):
        """
        Gets a copy of the list of ingredients that this recipe has.
        @return: A copy of the list of ingredients
        """
        to_ret = [i for i in self.__ingredients_list]
        return to_ret

    def has(self, ingredient):
        """
        Returns whether or not this recipe has the given ingredient.
        @param ingredient: An ingredient (str)
        @return: True if ingredient is present, False otherwise
        """
        return ingredient in self.__ingredients_list
