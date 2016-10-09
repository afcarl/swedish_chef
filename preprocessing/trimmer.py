"""
This preprocessing module provides all methods associated with
trimming the text files of the data.
"""

import preprocessing.prep_global as prep
import chef_global.debug as debug

def __trim_non_recipe(cookbook_file_path):
    """
    Takes a cookbook data file path and removes all the non-recipe, non-ingredient
    info from it.
    @param cookbook_file_path: A path to a cookbook data file (encoded in XML)
    @return: void
    """
    #TODO
    debug.debug_print("Attempting to trim " + str(cookbook_file_path))

def _trim_all_files_to_recipes():
    """
    Trims away all the non-recipe, non-ingredient stuff from the data files.
    @return: void
    """
    prep._apply_func_to_each_data_file(__trim_non_recipe)
