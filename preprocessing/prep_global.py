"""
This module contains all of the methods that the other preprocessor methods rely
on.
"""

import os
import chef_global.config as config
from chef_global.debug import assert_value_is_set
from chef_global.debug import debug_print


def __check_dir_has_xml(dir_path):
    """
    Checks whether the given directory contains any files
    that are xml files.
    @param dir_path: The path to the directory we want to check.
    @return: True if dir_path contains at least on xml file, False otherwise
    """
    if os.path.isdir(dir_path):
        for f_name in os.listdir(dir_path):
            if f_name.lower().endswith(".xml"):
                return True
        return False
    else:
        raise ValueError("dir_path needs a valid directory.")


def __get_arg_value(args, sub_module, sub_module_err_string):
    """
    Gets the argument value from the args, given which
    sub_module it is that is specified.
    @param args: The ArgParse args
    @param sub_module: The sub_module from _sanitize_input
    @param sub_module_err_string: The string to print in the case
                            of the sub_module being incorrect/unkown
    @return: the arg value
    """
    if sub_module == "TRIMMER":
        return args.trim
    elif sub_module == "MOVER":
        return args.reset[0]  # Get the master directory from the args
    else:
        print(sub_module_err_string)
        raise ValueError


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
    list_of_file_names = [file_name for
                            file_name in os.listdir(config.DATA_DIRECTORY)]

    for f in list_of_file_names:
        func(f)


def _sanitize_input(args, sub_module, key):
    """
    Validate the input args, but only those specified by key.
    If invalid, exit with a helpful error message.
    @param sub_module: one of ["TRIMMER", "MOVER"]
    @param key: one of ["DATA"] - "DATA" --> check if the sub_module value is
                                             a valid data directory
    @return: void
    """
    # First declare exit strings that may be used throughout this function
    sub_module_err_string = "Illegal code path -> " + \
                            "_sanitize_input requires one of [" + \
                            "'TRIMMER', 'MOVER'" + \
                            "]"
    key_err_string = "Illegal code path -> " + \
                     "_sanitize_input requires one of [" + \
                     "'DATA'" + \
                     "]"

    if key == "DATA":
        # If the key is DATA, we must be either the trimmer or the mover
        dir_path = __get_arg_value(args, sub_module, sub_module_err_string)

        # Now that we figured out that the directory is a real directory,
        # check to make sure it actually contains xml files
        valid_dir = __check_dir_has_xml(dir_path)
        if valid_dir:
            return  # Valid input
        else:
            print("You must specify a valid data directory." +
                  " A valid directory contains xml files of cookbooks.")
            exit(1)
    else:
        # If we got here, we did not specify a correct key
        print(key_err_string)
        raise ValueError







