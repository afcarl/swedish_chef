"""
This module provides the API functions for the preprocessing suite of tools
for the chef. ALL API functions for this package are in this file.
"""

import time
import os
from tqdm import tqdm
import myio.myio as myio
import random
import statistics.ingredients_table as ingredients_table
import preprocessing.prep_global as prep_global
import chef_global.debug as debug
import chef_global.config as config
import preprocessing.trimmer as trimmer
import preprocessing.mover as mover



def copy_master_to_data_location():
    """
    Copies the master directory files to the data directory and deletes all the ones
    that were in the data directory before the copy.
    @return: void
    """
    mover._copy_master_to_data_location()


def execute_commands(args):
    """
    Executes preprocessor commands from the command line arguments.
    Currently, args could contain several things for the preprocessor to do,
    but if so, those things will be done in an unspecified order.
    @param args: The arguments from the command line (ArgParser)
    @return: void
    """
    did_something = False

    if args.prep:
        __do_pipeline(args)
        did_something = True
    if args.tabulate:
        __do_tabulate_ingredients(args)
        did_something = True
    if args.trim:
        __do_trim(args)
        did_something = True
    if args.reset:
        __do_reset(args)
        did_something = True

    if not did_something:
        print("A command was sent to the preprocessor that it didn't recognize.")
        exit(-1)


def gather_random_recipes(recipe_file_path, num_recipes, seed=None):
    """
    Gathers num_recipes recipes from the file at recipe_file_path. The
    particular recipes gathered will be uniformly random with no replacement.
    An optional seed can be given to ensure the same recipes will be given
    each call, assuming the same file path.
    @param recipe_file_path: The path to the recipe file. The recipe file must
                             be already preprocessed to have its recipes separated
                             by the recipe separator from the config file.
    @param num_recipes: The number of recipes to give back
    @param seed: The random seed to use. Should be an integer or None.
    @return: A list of recipes, which are just strings of whatever happen
             to be between the file's recipe separator tags.
    """
    total_num_recipes = myio.count_occurrences(recipe_file_path, config.NEW_RECIPE_LINE.lower())

    if total_num_recipes < num_recipes:
        err_msg = "num_recipes cannot exceed the actual total number of recipes " +\
                  "in the recipe file given to gather_random_recipes. You passed in " +\
                  str(num_recipes) + ", but there are only " + str(total_num_recipes) +\
                  " recipes in the recipe file."
        raise ValueError(err_msg)
    elif total_num_recipes == num_recipes:
        debug.debug_print("num_recipes equals the total number of recipes in the " +\
                          "recipe file. This is a little irregular, as this means that " +\
                          "the 'random' recipes will actually just be ALL of the recipes.")
        recipe_indeces = [i for i in range(0, num_recipes + 1)]
    else:
        # Generate the recipe indeces that we should gather:
        print("    |-> Generating the indeces...")
        random.seed(seed)
        recipe_indeces = []
        while len(recipe_indeces) != num_recipes:
            next_index = None
            while next_index in recipe_indeces or next_index is None:
                next_index = random.randint(0, total_num_recipes - 1)
            recipe_indeces.append(next_index)

    recipes = []
    print("    |-> Retrieving those recipes...")
    for index in tqdm(recipe_indeces):
        recipe = trimmer._get_recipe_at_index(index, recipe_file_path)
        recipes.append(recipe)

    return recipes


def run_unit_tests():
    """
    Run the unit tests for the preprocessor.
    @return: void
    """
    trimmer._clean_ingredient_test()
    trimmer._remove_duplicates_between_bounds_test()
    trimmer._remove_plurals_test()


def trim_all_files_to_recipes():
    """
    Trims all of the data files so that they have only recipes in them (with ingredients) and
    XML tags, but no intros, chapter headings, etc.
    @return: void
    """
    debug.debug_print("Calling trim_all_files_to_recipes...")
    debug.assert_value_is_set(config.DATA_DIRECTORY, "config.DATA_DIRECTORY")
    trimmer._trim_all_files_to_recipes()


def __do_single_word():
    print("Replacing all ingredients in big recipe file with single word versions...")
    if os.path.exists(config.RECIPE_FILE_SINGLE_PATH):
        print("    |-> Found existing single word version at " + \
                str(config.RECIPE_FILE_SINGLE_PATH) + ", using that one.")
        pass
    else:
        print("    |-> Could not find existing version. Generating new one, this will " +\
                        "take a while. Started at: " + str(time.strftime("%I:%M:%S")))
        trimmer._replace_all_ingredients_with_single_words(
                                                config.RECIPE_FILE_PATH, config.UNIQUE)
 



def __do_pipeline(args):
    """
    Respond to the request to do the entire preprocessor pipeline.
    @param args: ArgParse object
    @return: void
    """
    __do_reset(args)
    __do_trim(args)
    __do_tabulate_ingredients(args)
    __do_single_word()
    __do_reset(args)


def __do_reset(args):
    """
    Respond to the request to reset data.
    Reset the data directory from the given master copy location.
    @param args: ArgParse object
    @return: void
    """
    print("Reseting files...")
    prep_global._sanitize_input(args, "MOVER", "DATA")
    if args.reset:
        config.MASTER_DATA_DIRECTORY = args.reset[0]
        config.DATA_DIRECTORY = args.reset[1]
    else:
        config.MASTER_DATA_DIRECTORY = args.prep[0]
        config.DATA_DIRECTORY = args.prep[1]
    debug.debug_print("Mover activated...")
    copy_master_to_data_location()


def __do_tabulate_ingredients(args):
    """
    Respond to the request to tabulate ingredients.
    Take each data file and combine it, then parse the combination
    down into the unique ingredients used by each recipe as well as
    a list of all the ingredients used by all the recipes.
    @param args: ArgParse data
    @return: void
    """
    prep_global._sanitize_input(args, "TRIMMER", "DATA")
    if args.tabulate:
        config.DATA_DIRECTORY = args.tabulate
    else:
        config.DATA_DIRECTORY = args.prep[1]
    print("Creating IngredientsTable and saving it to disk...")
    trimmer._prepare_tabulate_ingredients()
    ing_table = ingredients_table.IngredientsTable(config.UNIQUE)
    config.INGREDIENT_TABLE_PATH = ingredients_table.save_to_disk(ing_table)


def __do_trim(args):
    """
    Respond to the request to trim data.
    If the data directory is valid, trim the xml files down to their recipes.
    @param args: ArgParse object
    @return: void
    """
    print("Trimming the xml files to recipes...")
    prep_global._sanitize_input(args, "TRIMMER", "DATA")
    if args.trim:
        config.DATA_DIRECTORY = args.trim
    else:
        config.DATA_DIRECTORY = args.prep[1]
    debug.debug_print("Trimmer activated...")
    trim_all_files_to_recipes()

    # Now you have all of the recipes, but in different files
    # word2vec wants one big recipe file in plain English, so
    # make that and save it.
    print("Creating the big recipe file " + config.RECIPE_FILE_PATH)
    if os.path.exists(config.RECIPE_FILE_PATH):
        print("    |-> Found existing big recipe file at " + \
                str(config.RECIPE_FILE_PATH) + ", using that one.")
        pass
    else:
        print("    |-> Could not find existing version. Generating new one...")
        mover._append_all_recipe_files()

        print("Trimming big recipe file...")
        trimmer._remove_xml_from_file(config.RECIPE_FILE_PATH)

        print("Removing all new_recipe lines from the big recipe file...")
        myio.find_replace(config.RECIPE_FILE_PATH, config.NEW_RECIPE_LINE, "")

        print("Stripping big recipe file...")
        myio.strip_file(config.RECIPE_FILE_PATH)
    






