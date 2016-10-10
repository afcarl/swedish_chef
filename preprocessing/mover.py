"""
Module for moving files around for the preprocessor.
"""

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
    print("Moving data files from " + str(config.MASTER_DATA_DIRECTORY) +
            " to " + str(config.DATA_DIRECTORY))
