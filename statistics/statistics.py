"""
The main API for the statistics python package.
"""

import preprocessing.preprocessing as preprocessor
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy.sparse as sparse
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import pandas as pd
import numpy as np
import chef_global.debug as debug
import myio.myio as myio
import statistics.ingredients_table as it
import chef_global.config as config
from statistics.recipe import Recipe


def train_models():
    """
    Trains the models and saves them to disk.
    @param args: ArgParse object
    @return: void
    """
    it_path = args.math[0]
    table = it.load_from_disk(it_path)

    unique = args.math[1]
    unique_within = args.math[2]

    print("Generating recipes...")
    recipes = __generate_recipes(table, unique_within)

    row_clusters = __generate_linkage(recipes, table)

    print("Saving row_clusters...")
    myio.save_pickle(row_clusters, config.CLUSTERS)

#    This is, ludicrously, a recursive algorithm, so it stack overflows
#    print("Generating dendrogram...")
#    row_dendr = dendrogram(row_clusters, labels=labels)
#    print("Plotting it...")
#    plt.tight_layout()
#    plt.ylabel("Eucliden Distance")
#    plt.show()
    # TODO

def __generate_linkage(recipes, table):
    """
    Generate the hierarchical linkage matrix by the clustering algorithm.
    """
    print("Generating the variables column...")
    variables = ["Recipe " + str(i) for i in range(len(recipes))]

    print("Generating the labels column...")
    labels = [ingredient for ingredient in table.get_ingredients_list()]

    print("Retrieving scipy version of sparse matrix...")
    sparse_matrix = __retrieve_sparse_matrix(recipes, labels).tocoo()

    print("Retrieving dense representation of matrix...")
    matrix = __retrieve_matrix(sparse_matrix)

    print("Retrieving dataframe...")
    df = __retrieve_dataframe(matrix, variables, labels)

    print("Generating row_clusters (takes about 3 or 4 hours)...")
    row_clusters = linkage(pdist(df, metric="euclidean"), method="complete")

    return row_clusters



def __retrieve_dataframe(matrix, variables, labels):
    """
    Returns the given matrix as a dataframe or else finds one on the disk.
    """
    # Can't pickle an object this big. Use HDF5 instead if
    # important.
#    if os.path.isfile(config.DATA_FRAME):
#        df = myio.load_pickle(config.DATA_FRAME)
#        print("Found data frame.")
#    else:
#        print("Generating dataframe...")
#        df = pd.DataFrame(matrix, columns=variables, index=labels)
#        print("Saving dataframe...")
#        myio.save_pickle(df, config.DATA_FRAME)
    print("Generating dataframe...")
    df = pd.DataFrame(matrix, columns=variables, index=labels)
    return df


def __retrieve_matrix(sparse_matrix):
    """
    Retrieves a dense (numpy) representation of the sparse matrix.
    @param sparse_matrix: The matrix
    @return: The numpy array
    """
#    if os.path.isfile(config.MATRIX):
#        # This is too big on the disk - you might consider
#        # using HDF5 if this gets to be a problem
#        matrix = myio.load_pickle(config.MATRIX)
#        print("Found matrix.")
#    else:
#        print("Generating dense matrix...")
#        matrix = sparse_matrix.toarray()
#        myio.save_pickle(matrix, config.MATRIX)
    print("Generating dense matrix from sparse one...")
    matrix = sparse_matrix.toarray()
    return matrix


def __retrieve_sparse_matrix(recipes, ingredients):
    """
    Retrieves a sparse matrix representation of the recipes and ingredients.
    That is, retrieves a sparse matrix of the form:
        recipe 0    recipe 1    ...
    ing0   1           0        ...
    ing1   0           0        ...
    Either generates it from given args or else finds
    it on the disk.
    @param recipes: All of the recipes
    @param ingredients: All of the ingredients
    @return: the matrix
    """
    if os.path.isfile(config.MATRIX_SPARSE):
        sparse_matrix = myio.load_pickle(config.MATRIX_SPARSE)
        print("Found sparse matrix.")
    else:
        print("Generating sparse matrix...")
        print("  |-> Generating the rows, this may take a while...")
        rows = [__generate_ingredient_feature_vector_sparse(ingredient, recipes)
                    for ingredient in tqdm(ingredients)]
        print("  |-> Generating the matrix from the rows...")
        sparse_matrix = sparse.vstack(rows)
        print("Pickling the sparse matrix...")
        myio.save_pickle(sparse_matrix, config.MATRIX_SPARSE)
    return sparse_matrix


def run_unit_tests():
    """
    Runs the statistics package's unit tests.
    @return: void
    """
    it.unit_test()
    __training_test();


def __generate_ingredient_feature_vector(ingredient, recipes):
    """
    Generates a vector of the form:
    Recipe 0    Recipe 1    Recipe 2    ...
        0          1           0
    Where a 0 means the ingredient is not present in that recipe and a 1
    means it is.
    This means the return value is a list of the form [0, 1, 0, ...] where
    each index represents a recipe and the value at that index represents
    present or not.
    @param ingredient: The ingredient for which to generate a feature vector
    @param recipes: All of the recipes in the same order across multiple
                    calls to this function
    @return: The feature vector
    """
    to_ret = [1 if recipe.has(ingredient) else 0 for recipe in recipes]
    return to_ret


def __generate_ingredient_feature_vector_sparse(ingredient, recipes):
    """
    Does exactly the same thing as __generate_ingredient_feature_vector, but
    in a sparse format, specifically, it returns a csr_matrix.
    """
    #print("    |-> generating '" + str(ingredient) + "'...")
    list_form = __generate_ingredient_feature_vector(ingredient, recipes)
    lil_matrix_form = sparse.lil_matrix(list_form)
    return lil_matrix_form.tocsr()


def __generate_recipes(table, unique_within_path):
    """
    Generates a Recipe object from each recipe found in
    the file at unique_within_path, using the table to
    generate the IDs.
    @param table: The IngredientsTable to use
    @param unique_within_path: The path to the file containing the recipes
    @return: A list of recipe objects
    """
    to_ret = []
    recipe_producer = \
        myio.get_lines_between_tags(unique_within_path, config.NEW_RECIPE_LINE.lower())
    lines_between_tags = next(recipe_producer)
    while lines_between_tags is not None:
        ingredients = [line.rstrip() for line in lines_between_tags]
        recipe = Recipe(table, ingredients=ingredients)
        to_ret.append(recipe)
        debug.debug_print("Recipe Generated: " + str(recipe))
        try:
            lines_between_tags = next(recipe_producer)
        except StopIteration:
            lines_between_tags = None

    return to_ret


def __training_test():
    """
    Run the unit test for the training algorithms.
    @return: void
    """
    test_name = "Training Test"
    debug.print_test_banner(test_name, False)
    # Use the same random seed to collect some random recipes and run the
    # algorithm on those recipes.
    if config.UNIQUE_WITHIN == "":
        # You must first preprocess the cookbooks
        print("The training test could not be run, because there is no unique_within file.")
        print("Please generate a unique_within file by running the preprocessor pipeline.")
    else:
        num_recipes = 10
        random_recipes = \
            preprocessor.gather_random_recipes(config.UNIQUE_WITHIN, num_recipes, seed=245)

        # Check if the length of the random recipes is correct:
        if len(random_recipes) != num_recipes:
            print("Test 0: FAIL --> The length of the recipe list returned did not match " +\
                  "what was expected: Got len() == " + str(len(random_recipes)) +\
                  " expected: len() == " + str(num_recipes))
        else:
            print("Test 1: passed --> len of random recipes is what was expected.")

        # You should now have a list of lists of ingredients
        # Append the new recipe line to the end of each recipe before saving it
        tmp_recipes = []
        ingredients = []
        for recipe in random_recipes:
            for ingredient in recipe:
                ingredients.append(ingredient)
                tmp_recipes.append(ingredient)
            tmp_recipes.append(config.NEW_RECIPE_LINE.lower())

        ingredients = set(ingredients)

        print("Saving " + str(tmp_recipes))
        print("Ingredients: " + str(ingredients))

        # Save the temporary recipe file
        unique_within = "tmp/unique_within.TEST"
        myio.write_list_to_file(unique_within, tmp_recipes)

        # Save the temporary ingredients file
        unique = "tmp/unique.TEST"
        myio.write_list_to_file(unique, ingredients)

        # Next, make ingredient table from the recipes
        table = it.IngredientsTable(unique)

        # Now put the chosen recipes into a reasonable format
        recipes = __generate_recipes(table, unique_within)

        # Now generate the hierarchical linkage from the ingredients
        linkage = __generate_linkage(recipes, table)

        # Clean up
        os.remove(unique_within)
        os.remove(unique)

    debug.print_test_banner(test_name, True)





