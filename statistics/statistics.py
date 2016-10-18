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

    variables = ["Recipe " + str(i) for i in range(len(recipes))]
    labels = [ingredient for ingredient in table.get_ingredients_list()]
    myio.write_list_to_file("FEATURE_VECTOR____", recipes[3456].get_feature_vector())
    # TODO



def run_unit_tests():
    """
    Runs the statistics package's unit tests.
    @return: void
    """
    it.unit_test()


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









