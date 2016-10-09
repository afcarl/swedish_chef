"""
This module contains all of the methods that the other preprocessor methods rely
on.
"""

import os
import chef_global.config as config
from chef_global.debug import assert_value_is_set
from chef_global.debug import debug_print

def apply_func_to_each_data_file(func):
    """
    Applies the given function to each data file in the cookbook data
    directory.
    @param func: A function which takes a file path as an input (and
                 presumably, does something with the file)
    @return: void
    """
    debug_print("Applying a function to each file in the data directory...")

    # Collect each file in the directory
    assert_value_is_set(config.DATA_DIRECTORY, "config.DATA_DIRECTORY")
    list_of_file_names = [file_name for file_name in os.listdir(config.DATA_DIRECTORY)]

    for f in list_of_file_names:
        func(f)
