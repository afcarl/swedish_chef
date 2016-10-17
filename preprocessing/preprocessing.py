"""
This module provides the API functions for the preprocessing suite of tools
for the chef. ALL API functions for this package are in this file.
"""

from statistics.IngredientsTable import IngredientsTable
import preprocessing.prep_global as prep_global
import chef_global.debug as debug
import chef_global.config as config
import preprocessing.trimmer as trimmer
import preprocessing.mover as mover


def __do_pipeline(args):
    """
    Respond to the request to do the entire preprocessor pipeline.
    @param args: ArgParse object
    @return: void
    """
    __do_reset(args)
    __do_trim(args)
    __do_tabulate_ingredients(args)


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
    Then do some basic statistics on the ingredients and make an
    IngredientsTable object out of them.
    @param ArgParse data
    @return: void
    """
    print("Tabulating ingredients...")
    prep_global._sanitize_input(args, "TRIMMER", "DATA")
    if args.tabulate:
        config.DATA_DIRECTORY = args.tabulate
    else:
        config.DATA_DIRECTORY = args.prep[1]
    trimmer._prepare_tabulate_ingredients()
    ing_table = IngredientsTable(config.UNIQUE)
    config.INGREDIENT_TABLE_PATH = ing_table.save_to_disk()


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


