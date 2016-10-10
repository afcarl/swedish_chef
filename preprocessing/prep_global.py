"""
This module contains all of the methods that the other preprocessor methods rely
on.
"""

import os
import chef_global.config as config
from chef_global.debug import assert_value_is_set
from chef_global.debug import debug_print

def _apply_func_to_each_data_file(func):
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


def _sanitize_input(args, sub_module, key):
    """
    Validate the input args, but only those specified by key.
    If invalid, exit with a helpful error message.
    @param sub_module: one of ["TRIMMER"] - "TRIMMER" --> The trimmer
    @param key: one of ["DATA"] - "DATA" --> check if the sub_module value is a valid data directory
    @return: void
    """
    if key == "DATA":
        if sub_module == "TRIMMER":
            dir_path = args.trim
        else:
            print("Illegal code path -> _sanitize_input requires one of ['TRIMMER']")
            raise ValueError

        if os.path.isdir(dir_path):
            # The dir_path is actually a directory, now check if it contains xml files
            # If there is at least one xml file in it, we will consider the dir valid
            for f_name in os.listdir(dir_path):
                if f_name.lower().endswith(".xml"):
                    return  # This is a valid argument
            print("You must specify a valid data directory with trimmer. A valid directory contains xml files of cookbooks.")
            exit(1)
        else:
            print("You must specify a valid data directory with trimmer. A valid directory contains xml files of cookbooks.")
            exit(1)
    else:
        print("Illegal code path -> _sanitize_input requires one of ['DATA']")
        raise ValueError







