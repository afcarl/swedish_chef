"""
This module holds debug functionality for the Swedish Chef, most especially debug_print.
"""

import os
import chef_global.config

def assert_value_is_set(value, name_of_value):
    """
    Asserts that the given value is not None and is not an
    empty string. If the assertion fails, an explanation is
    printed and the program exits.
    @param value: The value to check
    @param name_of_value: The name of the value to check
    @return: void
    """
    if value is None or value == "":
        print("ASSERTION FAILED:" + os.linesep + name_of_value + " is not set.")
        exit(-1)
    else:
        debug_print("Assertion passed: " + name_of_value + " is set.")

def debug_print(message):
    """
    Prints the given message, but only if the verbosity level allows it.
    """
    if chef_global.config.VERBOSE:
        print(message)
