"""
This module provides the API functions for the preprocessing suite of tools
for the chef. ALL API functions for this package are in this file.
"""

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
    @param args: The arguments from the command line (ArgParser)
    @return: void
    """
    if args.trim:
        # If the data directory is valid, trim the xml files down to their recipes
        prep_global._sanitize_input(args, "TRIMMER", "DATA")
        config.DATA_DIRECTORY = args.trim
        debug.debug_print("Trimmer activated...")
        trim_all_files_to_recipes()
    if args.reset:
        # Reset the data directory from the given master copy location
        prep_global._sanitize_input(args, "MOVER", "DATA")
        config.DATA_DIRECTORY = args.reset[1]
        config.MASTER_DATA_DIRECTORY = args.reset[0]
        debug.debug_print("Mover activated...")
        copy_master_to_data_location()

def trim_all_files_to_recipes():
    """
    Trims all of the data files so that they have only recipes in them (with ingredients) and
    XML tags, but no intros, chapter headings, etc.
    @return: void
    """
    debug.debug_print("Calling trim_all_files_to_recipes...")
    debug.assert_value_is_set(config.DATA_DIRECTORY, "config.DATA_DIRECTORY")
    trimmer._trim_all_files_to_recipes()
