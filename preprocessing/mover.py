"""
Module for moving files around for the preprocessor.
"""

import os
import shutil
import preprocessing.prep_global as prep_global
import chef_global.debug as debug
import chef_global.config as config

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
        debug.debug_print("Copying " + f_name + " to new directory...")
        shutil.copyfile(f, os.path.join(config.DATA_DIRECTORY, f_name))







