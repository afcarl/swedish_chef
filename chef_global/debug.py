"""
This module holds debug functionality for the Swedish Chef, most especially debug_print.
"""

import chef_global.config

def debug_print(message):
    """
    Prints the givene message, but only if the verbosity level allows it.
    """
    if chef_global.config.VERBOSE:
        print(message)
