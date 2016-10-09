"""
This module provides the API functions for the preprocessing suite of tools
for the chef. ALL API functions for this package are in this file.
"""

import preprocessing.prep_global as prep_global
import chef_global.debug as debug
import chef_global.config as config


def trim_all_files_to_recipes():
    """
    Trims all of the data files so that they have only recipes in them (with ingredients) and
    XML tags, but no intros, chapter headings, etc.
    @return: void
    """
    debug.debug_print("Calling trim_all_files_to_recipes...")
    debug.assert_value_is_set(config.DATA_DIRECTORY, "config.DATA_DIRECTORY")
    trimmer._trim_all_files_to_recipes()
