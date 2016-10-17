"""
Module to hold global configurations for the Swedish Chef.
Some of these are marked as to be changed by hand, while some
are marked to be changed programmatically. Please pay attention.
"""

# The directory wherein we can find the data for the chef
# @programmatic
DATA_DIRECTORY = None

# The path to the saved IngredientsTable object
# @programmatic
INGREDIENT_TABLE_PATH = None

# The master directory where we store data
# @programmatic
MASTER_DATA_DIRECTORY = None

# The unique.txt ingredient file path
# @programmatic
UNIQUE = ""

# The unique_within.txt ingredient file path
# @programmatic
UNIQUE_WITHIN = ""

# The unit test banner to print around a unit test start and ending
# @manual
UNIT_TEST_HEADER = "============================================="

# Whether debug output should be printed as the program executes
# @programmatic
VERBOSE = False
