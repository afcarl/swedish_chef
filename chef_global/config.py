"""
Module to hold global configurations for the Swedish Chef.
Some of these are marked as to be changed by hand, while some
are marked to be changed programmatically. Please pay attention.
"""

# Clusters
# @manual
CLUSTERS = "tmp/clusters"

# The directory wherein we can find the data for the chef
# @programmatic/manual - can be overwritten by user, but default is manually set
DATA_DIRECTORY = "tmp"

# The dataframe
# @manual
DATA_FRAME = "tmp/dataframe"

# The path to the saved IngredientsTable object
# @programmatic
INGREDIENT_TABLE_PATH = None

# The master directory where we store data
# @programmatic
MASTER_DATA_DIRECTORY = None

# The path to the dense matrix
# @manual
MATRIX = "tmp/matrix"

# The path to the sparse data matrix
# @manual
MATRIX_SPARSE = "tmp/sparsematrix"

# The new cookbook marker
# @manual
NEW_COOKBOOK_LINE = "NEW_COOKBOOK______________________________LINE"

# The new recipe marker
# @manual
NEW_RECIPE_LINE = "NEW_RECIPE__________________________________LINE"

# The name of the file that will contain all of the recipes in human
# readable format.
# @manual
RECIPE_FILE_NAME = "all_of_the_recipes.txt"

# The path to the recipe file
# @programmatic
RECIPE_FILE_PATH = DATA_DIRECTORY + "/" + RECIPE_FILE_NAME

# The path to the recipe file with ingredients replaced with single word versions
# @programmatic
RECIPE_FILE_SINGLE_PATH = DATA_DIRECTORY + "/" + "single_" + RECIPE_FILE_NAME

# The unique.txt ingredient file path
# @programmatic/manual - will overwrite with user input
UNIQUE = "tmp/unique.txt"

# The unique_within.txt ingredient file path
# @programmatic/manual - will overwrite with user input
UNIQUE_WITHIN = "tmp/unique_within.txt"

# The unit test banner to print around a unit test start and ending
# @manual
UNIT_TEST_HEADER = "============================================="

# Whether debug output should be printed as the program executes
# @programmatic
VERBOSE = False

# The path to the word2vec model
# @manual
WORD2VEC_MODEL_PATH = "tmp/word2vec.model"
