"""
The main API for the statistics python package.
"""

import pandas as pd
import numpy as np
import chef_global.debug as debug
import myio.myio as myio
import statistics.ingredients_table as it
import chef_global.config as config
from statistics.recipe import Recipe


def calculate_stats(args):
    """
    Calculates some basic statistics using the given ingredients table
    and the unique and unique_within files given.
    @param args: ArgParse object
    @return: void
    """
    it_path = args.math[0]
    table = it.load_from_disk(it_path)

    unique = args.math[1]
    unique_within = args.math[2]

    print("Generating recipes...")
    recipes = __generate_recipes(table, unique_within)

    print("Generating the variables column...")
    variables = ["Recipe " + str(i) for i in range(len(recipes))]

    print("Generating the labels column...")
    labels = [ingredient for ingredient in table.get_ingredients_list()]

    print(str(__generate_ingredient_feature_vector("tomato", recipes)))

    #print("Generating the data matrix...")
    #data_matrix = [__generate_ingredient_feature_vector(ingredient, recipes)
    #                    for ingredient in labels]

    #print("Generating the data frame...")
    #df = pd.DataFrame(data_matrix, columns=variables, index=labels)
    # TODO



def run_unit_tests():
    """
    Runs the statistics package's unit tests.
    @return: void
    """
    it.unit_test()


def __generate_ingredient_feature_vector(ingredient, recipes):
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
    @param recipes: All of the recipes in the same order across multiple
                    calls to this function
    @return: The feature vector
    """
    print("    |-> generating '" + str(ingredient) + "'...")
    to_ret = [1 if recipe.has(ingredient) else 0 for recipe in recipes]
    return to_ret


def __generate_recipes(table, unique_within_path):
    """
    Generates a Recipe object from each recipe found in
    the file at unique_within_path, using the table to
    generate the IDs.
    @param table: The IngredientsTable to use
    @param unique_within_path: The path to the file containing the recipes
    @return: A list of recipe objects
    """
    to_ret = []
    recipe_producer = \
        myio.get_lines_between_tags(unique_within_path, config.NEW_RECIPE_LINE.lower())
    lines_between_tags = next(recipe_producer)
    while lines_between_tags is not None:
        ingredients = [line.rstrip() for line in lines_between_tags]
        recipe = Recipe(table, ingredients=ingredients)
        to_ret.append(recipe)
        debug.debug_print("Recipe Generated: " + str(recipe))
        try:
            lines_between_tags = next(recipe_producer)
        except StopIteration:
            lines_between_tags = None

    return to_ret









