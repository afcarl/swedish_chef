"""
Module for moving files around for the preprocessor.
"""

import glob
import os
import shutil
import myio.myio as myio
import preprocessing.prep_global as prep_global
import chef_global.debug as debug
import chef_global.config as config


def _append_all_recipe_files():
    """
    Takes each file in the config.DATA_DIRECTORY folder and
    appends them onto a single recipe file which will then
    contain plain English recipes all in one file.
    @return: void
    """
    if not os.path.isdir(config.DATA_DIRECTORY):
        raise ValueError("config.DATA_DIRECTORY not set.")
    else:
        files = glob.glob(config.DATA_DIRECTORY + "/*.xml")
        myio.join_all_files(files, config.RECIPE_FILE_PATH)


def _copy_master_to_data_location():
    """
    Copies the master directory files over the data
    directory. Deletes any files in the data directory
    before copying.
    @return: void
    """
    debug.debug_print("_copy_master_to_data_location called, making assertions...")
    debug.assert_value_is_set(config.MASTER_DATA_DIRECTORY,
                        "config.MASTER_DATA_DIRECTORY")
    debug.assert_value_is_set(config.DATA_DIRECTORY,
                        "config.DATA_DIRECTORY")

    print("Deleting old data files...")
    prep_global._apply_func_to_each_data_file(os.remove)

    print("Copying data files from " + str(config.MASTER_DATA_DIRECTORY) +
            " to " + str(config.DATA_DIRECTORY))
    # Collect files to copy
    list_of_file_names = [os.path.join(config.MASTER_DATA_DIRECTORY, file_name) for
                          file_name in os.listdir(config.MASTER_DATA_DIRECTORY)]

    for f in list_of_file_names:
        f_name = os.path.split(f)[-1]
        if f_name != "zuni.xml":
            # Don't include zuni - it is super weird
            debug.debug_print("Copying " + f_name + " to new directory...")
            shutil.copyfile(f, os.path.join(config.DATA_DIRECTORY, f_name))







