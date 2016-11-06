"""
The main API for the statistics python package.
"""

import random
import pandas
import statistics.recipe_table as recipe_table
import statistics.similar as similar
from sklearn.cluster import KMeans
import string
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import time
import preprocessing.preprocessing as preprocessor
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
import warnings
import statistics.cluster as cluster
import statistics.recipe_generator as recipe_generator
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import gensim
import matplotlib
try:
    if os.environ["SSH_CONNECTION"]:
        matplotlib.use("Pdf")
except KeyError:
    pass
import matplotlib.pyplot as plt

def ask_similar(args):
    """
    Uses any already trained models to figure out the
    similarity between the given list of ingredients, prints
    the similarity matrix, and then gives args.n number
    of ingredients that are also similar to the given
    list of ingredients.
    @param args: ArgParse object that must have args.similar[0] (number
                 of ingredients to get back) and args.simliar[1] (the
                 list of ingredients to compute a matrix for and to
                 match new ingredients with - may be an empty list,
                 in which case the program simply gives back a
                 list of similar ingredients of len args.similar[0])
    @return: void
    """
    # Get the number of ingredients the user wants and the ingredients they want to include
    num_ingredients = int(args.similar[0])
    generate_a_recipe = True if args.similar[1].strip().lower() == "y" else False
    ingredients = args.similar[2:]
    print("Ingredients: " + str(ingredients))

    # Load models and datastructures needed
    rec_table = recipe_table.load_from_disk(config.RECIPE_TABLE_PATH)
    clusters = cluster.load_clusters()
    rec_table.load_in_clusters(clusters)

    # Make sure all the ingredients the user asked to include actually exist
    for ingredient in ingredients:
        ingredient_exists = __check_for_ingredient(ingredient)
        if not ingredient_exists:
            print("Ingredient: " + str(ingredient) + " not found in models. Please replace.")
            return

    # If the user asked for zero ingredients, they want a random amount of ingredients
    # Get a random number, centered around the average number of ingredients in a recipe
    # Don't use anything less than three
    num_ingredients = rec_table.get_random_number()\
                        if num_ingredients is 0 else num_ingredients
    num_ingredients = 3 if num_ingredients < 3 else num_ingredients

    # Do what the user asked (get all new ingredients or get similar ones to given ones)
    if len(ingredients) is 0:
        ingredients = similar._get_random_similar_ingredients(num_ingredients, rec_table)
    else:
        sims = similar._get_similar_ingredients_to(ingredients, num_ingredients, rec_table)
        ingredients.extend(sims)

    if ingredients is None:
        print("Could not get that many random similar ingredients. Maybe just try again.")
    elif len(ingredients) is 1:
        print("Here is your random ingredient: " + str(ingredients[0]))
    else:
        print("Here are: " + str(num_ingredients) + " similar ingredients: ")
        print(str(ingredients))

        similarity_matrix, similarity_score, similarity_measure = \
                                similar._compute_similarity_stats(ingredients)

        print(str(pandas.DataFrame(similarity_matrix,
                                columns=ingredients, index=ingredients)))
        print("Similarity score for these ingredients: " + str(similarity_score))
        print("Z-score for similarity: " + str(similarity_measure))

    # If the user wants to generate a recipe, use the ingredients in that recipe
    if generate_a_recipe:
        recipe_generator._generate_recipe(ingredients, rec_table)



def train_models(args):
    """
    Trains the models and saves them to disk.
    @param args: ArgParse object
    @return: void
    """
    it_path = args.train[0]
    table = it.load_from_disk(it_path)

    unique = args.train[1]
    unique_within = args.train[2]

    print("Generating recipes...")
    rec_table = __generate_recipes(table, unique_within)
    myio.save_pickle(rec_table, config.RECIPE_TABLE_PATH)
    recipes = rec_table.get_recipes()
    print("Length of recipe table: " + str(len(recipes)))

    print("Generating the data structures...")
    variables, labels, sparse_matrix, matrix =\
                        __generate_model_structures(rec_table, table)

    # If we ran out of memory, matrix is none. We must use the sparse matrix
    # to regenerate the matrix on the fly from now on

    print("Running word2vec on recipes...")
    vec_model = __train_word2vec(table, recipes)

    print("Clustering using kmeans...")
    kmeans = __train_kmeans(matrix, sparse_matrix)

    print("Regenerating all of the kmeans clusters and saving them to disk. This will take a while...")
    clusters = cluster.regenerate_clusters(kmeans, rec_table)

    print("Using a compression algorithm to encode training data...")
    recipe_generator._encode_training_data(rec_table)

    print("Training the RNN and saving it...")
    recipe_generator._train_rnn(rec_table)

    print("Computing statistics on the data...")
    __compute_stats(rec_table)

#    row_clusters = __generate_linkage(recipes, table)

#    print("Saving row_clusters...")
#    myio.save_pickle(row_clusters, config.CLUSTERS)

#    This is, ludicrously, a recursive algorithm, so it stack overflows
#    print("Generating dendrogram...")
#    row_dendr = dendrogram(row_clusters, labels=labels)
#    print("Plotting it...")
#    plt.tight_layout()
#    plt.ylabel("Eucliden Distance")
#    plt.show()
    # TODO


def __gather_all_ingredients(unique_within_path):
    """
    Gathers all the ingredients into a list of the form:
    [[ingredients from recipe 1], [ingredients from recipe 2], etc.]
    @param unique_within_path: The path to the unique within text file.
    @return: list of ingredients lists
    """
    print("        |-> Gathering all of the ingredients...")
    ingredients_producer= \
        myio.get_lines_between_tags(unique_within_path, config.NEW_RECIPE_LINE.lower())
    lines_between_tags = next(ingredients_producer)
    list_of_ingredients_lists = []
    while lines_between_tags is not None:
        ingredients = [line.rstrip().replace(" ", "_") for line in lines_between_tags]
        debug.debug_print("        |-> Ingredients from this iteration: " +\
                          str(ingredients))
        list_of_ingredients_lists.append(ingredients)
        try:
            lines_between_tags = next(ingredients_producer)
        except StopIteration:
            lines_between_tags = None
    return list_of_ingredients_lists


def __gather_all_recipe_texts():
    """
    Gathers all of the recipes' texts into a list of the form:
    ["text from recipe 1", "text from recipe 2", etc.]
    @return: A list of recipe texts.
    """
    print("        |-> Gathering all of the recipe steps...")
    to_ret = []
    recipe_producer=\
        myio.get_lines_between_tags(\
                config.RECIPE_FILE_SINGLE_PATH, config.NEW_RECIPE_LINE)
    lines_between_tags = next(recipe_producer)
    list_of_recipe_texts= []
    while lines_between_tags is not None:
        recipe = [line.rstrip() for line in lines_between_tags]
        debug.debug_print("        |-> Recipe from this iteration: " +\
                            str(recipe))
        list_of_recipe_texts.append(recipe)
        try:
            lines_between_tags = next(recipe_producer)
        except StopIteration:
            lines_between_tags = None
    return list_of_recipe_texts


def __generate_model_structures(rec_table, table, testing=False):
    """
    Generates all the necessary data structures for training the models.
    @param rec_table: A RecipeTable object
    @param table: An IngredientsTable object containing all the ingredients
                  found in the list of recipes.
    @param testing: Whether we are just testing the data
    @return: The datastructures
    """
    recipes = rec_table.get_recipes()

    print("    |-> Generating the variables heading...")
    variables = ["Recipe " + str(i) for i in range(len(recipes))]
    debug.debug_print("Variables: " + os.linesep + str(variables))

    print("    |-> Generating the labels column (takes a moment)...")
    labels = [ingredient for ingredient in table.get_ingredients_list()]
    debug.debug_print("Labels: " + os.linesep + str(labels))

    print("    |-> Retrieving scipy version of sparse matrix...")
    sparse_matrix = __retrieve_sparse_matrix(rec_table, labels, testing).tocoo()
    debug.debug_print("Sparse matrix: " + os.linesep + str(sparse_matrix))

    print("    |-> Retrieving dense representation of matrix...")
    matrix = __retrieve_matrix(sparse_matrix, testing)
    debug.debug_print("Dense matrix: " + os.linesep + str(pd.DataFrame(matrix)))

    return variables, labels, sparse_matrix, matrix


def __generate_linkage(recipes, table, matrix, testing=False):
    """
    Generate the hierarchical linkage matrix by the clustering algorithm.
    @param recipes: A list of recipe objects
    @param table: An IngredientsTable object containing all the ingredients
                  found in the list of recipes.
    @param matrix: A dense representation of the data matrix.
    @param testing: Whether to use any files found on disk/overwrite
                    those files. If not, temporary ones will be created.
    """
#    print("Normalizing row vectors...")
#    # Normalize the row vector (ingredients), so that they
#    # all have the same length, which means that ones that
#    # are in all kinds of recipes will have a much smaller
#    # score in any particular dimension, whereas those
#    # that only show up in a couple of recipes will have very
#    # strong scores in those
#    normalized_matrix = __normalize_rows(matrix)
#    debug.debug_print("Normalized matrix: " + os.linesep + str(pd.DataFrame(normalized_matrix)))
    normalized_matrix = matrix

    print("Running PCA on the matrix to reduce dimensionality...")
    matrix_after_pca = __run_pca(normalized_matrix)
    debug.debug_print("Matrix after PCA: " + os.linesep + str(pd.DataFrame(matrix_after_pca)))
#    matrix_after_pca = normalized_matrix

#    print("Now scaling the row vectors so that they aren't tiny numbers...")
#    # multiply each vector by like a thousand or something to make for reasonably
#    # sized numbers
#    scale_factor = 1000
#    scaled_matrix = matrix_after_pca * scale_factor
#    debug.debug_print("Scaled matrix: " + os.linesep + str(pd.DataFrame(scaled_matrix)))
    scaled_matrix = matrix_after_pca

    print("Generating the variables heading...")
    sm_rows = [row for row in scaled_matrix]
    variables = ["PCA comp " + str(i) for i in range(len(sm_rows[0].getA1()))]
    debug.debug_print("Variables: " + os.linesep + str(variables))

    print("Retrieving dataframe...")
    df = __retrieve_dataframe(scaled_matrix, variables, labels, testing)
    debug.debug_print("Data frame: " + os.linesep + str(df))

    print("Generating row_clusters (takes about 3 or 4 hours)...")
    print("Started at " + myio.print_time())
    row_clusters = linkage(pdist(df, metric="jaccard"), method="ward")

    return row_clusters, labels


def __normalize_rows(m):
    """
    Takes each row from the matrix and normalizes them into a vector
    of unit length.
    @param m: The matrix to normalize
    @return: The matrix, but with each row normalized.
    """
    normalized_matrix = []
    ma = np.matrix(m)
    for i, row in enumerate(ma):
        row_flat = row.getA1()
        if np.count_nonzero(row_flat) == 0:
            normalized_row = row_flat
        else:
            normalized_row = row_flat / np.linalg.norm(row_flat)
        normalized_matrix.append(normalized_row)

    return np.matrix(normalized_matrix)


def __retrieve_dataframe(matrix, variables, labels, testing=False):
    """
    Returns the given matrix as a dataframe or else finds one on the disk.
    """
    # Can't pickle an object this big. Use HDF5 instead if
    # important.
#    if not testing and os.path.isfile(config.DATA_FRAME):
#        df = myio.load_pickle(config.DATA_FRAME)
#        print("Found data frame.")
#    else:
#        print("Generating dataframe...")
#        df = pd.DataFrame(matrix, columns=variables, index=labels)
#        if not testing:
    #        print("Saving dataframe...")
    #        myio.save_pickle(df, config.DATA_FRAME)
    print("Generating dataframe...")
    df = pd.DataFrame(matrix, columns=variables, index=labels)
    return df


def __retrieve_matrix(sparse_matrix, testing=False):
    """
    Retrieves a dense (numpy) representation of the sparse matrix.
    @param sparse_matrix: The matrix
    @param testing: If True, generate a new matrix and don't save it.
    @return: The numpy array
    """
#    if not testing and os.path.isfile(config.MATRIX):
#        # This is too big on the disk - you might consider
#        # using HDF5 if this gets to be a problem
#        matrix = myio.load_pickle(config.MATRIX)
#        print("Found matrix.")
#    else:
#        print("Generating dense matrix...")
#        matrix = sparse_matrix.toarray()
#        if not testing:
#            myio.save_pickle(matrix, config.MATRIX)
    try:
        print("        |-> Generating dense matrix from sparse one...")
        matrix = sparse_matrix.toarray()
        return matrix
    except MemoryError:
        print("        |-> Out of memory, working around this.")
        return None


def __retrieve_sparse_matrix(rec_table, ingredients, testing=False):
    """
    Retrieves a sparse matrix representation of the recipes and ingredients.
    That is, retrieves a sparse matrix of the form:
        recipe 0    recipe 1    ...
    ing0   1           0        ...
    ing1   0           0        ...
    Either generates it from given args or else finds
    it on the disk.
    @param rec_table: All of the recipes as a RecipeTable object
    @param ingredients: All of the ingredients
    @param testing: Whether to find a matrix on the disk or generate a tmp one.
                    True to generate a tmp one.
    @return: the matrix
    """
    if not testing and os.path.isfile(config.MATRIX_SPARSE):
        sparse_matrix = myio.load_pickle(config.MATRIX_SPARSE)
        print("        |-> Found sparse matrix.")
    else:
        print("        |-> Generating sparse matrix...")
        print("            |-> Generating the rows, this may take a while...")
        rows = [sparse.coo_matrix(rec_table.ingredient_to_feature_vector(ingredient))
                    for ingredient in tqdm(ingredients)]
        print("            |-> Generating the matrix from the rows...")
        sparse_matrix = sparse.vstack(rows)
        if not testing:
            print("            |-> Pickling the sparse matrix...")
            myio.save_pickle(sparse_matrix, config.MATRIX_SPARSE)
    return sparse_matrix



def __sparse_matrix_test():
    """
    Run the unit test for the sparse matrix algorithm.
    """
    test_name = "Sparse Matrix Test"
    debug.print_test_banner(test_name, False)

    table = it.load_from_disk("tmp/ingredient_table")

    ingredients = ["apple", "pear", "salt", "pepper", "pork", "chocolate", "caramel"]
    recipes = [Recipe(table, ingredients=["apple", "coffee"]),
               Recipe(table, ingredients=["salt, apple", "pork"]),
                Recipe(table, ingredients=["caramel"])]
    rec_table = recipe_table.RecipeTable(recipes=recipes)
    sm = __retrieve_sparse_matrix(rec_table, ingredients, testing=True)
    print("Generated this sparse matrix: " + str(sm))

    debug.print_test_banner(test_name, True)




def __run_pca(m):
    """
    Runs PCA on the given matrix and returns a matrix
    with fewer features (recipes), but which still
    captures most of the variance.
    @param m: The matrix before PCA
    @return: The matrix after PCA
    """
    captured_variance = 0.9  # Capture 90% of the variance
    pca = PCA(n_components=captured_variance, svd_solver="full")
    m_pca = pca.fit_transform(m)

    return np.matrix(m_pca)


def run_unit_tests():
    """
    Runs the statistics package's unit tests.
    @return: void
    """
    it.unit_test()
    __normalize_rows_test()
    __training_test();
    __sparse_matrix_test()
    similar._unit_test()


def __check_for_ingredient(ingredient):
    """
    Checks whether the given ingredient is found in the word2vec model (which
    doesn't contain all of the ingredients).
    @param ingredient: The ingredient of interest
    @return: True if word2vec has the ingredient, False otherwise
    """
    try:
        # Check the similarity against water, and if it throws an exception,
        # it is because it is not in the model
        similar._compute_similarity_stats([ingredient, "water"])
        return True
    except KeyError:
        return False



def __compute_stats(rec_table):
    """
    Computes the similarity mean and similarity standard
    deviation and maybe some other stats from the recipes.
    @param rec_table: The RecipeTable object that contains
                      all of the recipes.
    @return: void (just prints the information)
    """
    similar._compute_sim_stats(rec_table)


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
    @return: A RecipeTable object
    """
    if os.path.exists(config.RECIPE_TABLE_PATH):
        print("    |-> Found existing recipe table at " + str(config.RECIPE_TABLE_PATH) +\
              ", using that one...")
        return recipe_table.load_from_disk(config.RECIPE_TABLE_PATH)
    else:
        print("    |-> Generating a recipe table...")
        list_of_ingredients_lists = __gather_all_ingredients(unique_within_path)
        list_of_recipe_texts = __gather_all_recipe_texts()

        # KMeans needs to be retrained if use this
        #list_of_ingredients_lists = [l for l in list_of_ingredients_lists if len(l) > 0]
        #list_of_recipe_texts = [t for t in list_of_recipe_texts if t != ""]

        recipes = __generate_recipes_from_lists(\
                    list_of_ingredients_lists, list_of_recipe_texts, table)
        print("        |-> Creating a table of recipes from the data...")
        rec_table = recipe_table.RecipeTable(recipes)
        print("        |-> Saving the recipe_table at " + str(config.RECIPE_TABLE_PATH))
        recipe_table.save_to_disk(rec_table, config.RECIPE_TABLE_PATH)
        return rec_table

def __generate_recipes_from_lists(ingredients, texts, table):
    """
    Generates a list of Recipe objects, using the given
    ingredients and texts. Tries to do so intelligently
    so that the texts make sense with the ingredients.
    @param ingredients: A list of the form:
                        [[ingredients from recipe 1], [..rec 2], ...]
    @param texts: A list of the form:
                  ["recipe text 1", "recipe text 2", ...]
    @param table: An IngredientsTable object with ingredients from ingredients
    @return: A list of Recipe objects.
    """
#=================WORKS=======================#
#    recipes = []
#    for index, _ in enumerate(ingredients):
#        ings = ingredients[index]
#        #TODO: rather than using this line to get the text, you should
#        #      search through the recipes until you find one that has a majority
#        #      of the ingredients in it. Use that recipe for it and add it to
#        #      a list of back up recipes. If you can't find a recipe, find the
#        #      first recipe in the back ups that matches at least two of the ingredients
#        #      (if there are two ingredients, otherwise one) and use that one. Then
#        #      shuffle the backup list
#        t = "" if index >= len(ingredients) else texts[index]
#        recipes.append(Recipe(table, ingredients=ings, text=t))
#    return recipes
#=================END=======================#
    the_texts = []
    for t in texts:
        st = ""
        for line in t:
            st += line
        the_texts.append(st)

    recipes = []
    backup = []
    batches_to_check_later = []
    for time in range(2):
        if time is 0:
            print("            |-> Doing it the first time of two...")
            to_iterate = list(ingredients)
            print("            |-> This time list length: " + str(len(to_iterate)))
        else:
            print("            |-> Doing it the second time of two...")
            to_iterate = batches_to_check_later
            print("            |-> This time list length: " + str(len(to_iterate)))

        for batch in tqdm(to_iterate):
            found, recipe_text = __check_for_batch_in_list(batch, the_texts)
            if found:
                # We found what we believe is the right recipe in the list of
                # recipes.
                # Now that we've used it, move it from the list to a backup
                # list
                backup.append(recipe_text)
                the_texts.remove(recipe_text)
                recipe = Recipe(table, ingredients=batch, text=recipe_text)
                recipes.append(recipe)
            else:
                # We failed to find a recipe that has the majority of ingredients
                # from this ingredient batch in the normal list of recipes
                # This may be because we moved the recipe text to the backup list
                # So check that list
                found_in_backup, recipe_text = __check_for_batch_in_list(batch, backup)
                if not found_in_backup:
                    # We still couldn't find it, so it could be that this recipe text
                    # says something like "see recipe for foo", where foo is something
                    # that we haven't seen yet (and therefore isn't in the backup list)
                    # So add this batch to the list to check later
                    if time is 0:
                        batches_to_check_later.append(batch)
                    else:
                        # Just give up
                        recipe = Recipe(table, ingredients=batch, text="")
                        recipes.append(recipe)
                else:
                    # We found it in the backup list, so use it
                    recipe = Recipe(table, ingredients=batch, text=recipe_text)
                    recipes.append(recipe)

    return recipes


def __check_for_batch_in_list(batch, list_to_search):
    """
    Searches for a recipe text in list_to_search that contains a majority
    of the ingredients in batch. If it finds one, it returns it
    along with True. Else, False and None.
    @param batch: The batch of ingredients to search for
    @param list_to_search: The list of recipe texts to search through
    @return: A tuple: (True if found, recipe text if found otherwise None)
    """
    for recipe_text in list_to_search:
        majority = int((len(batch) / 2.0) + 0.5)
        num_found = 0
        for ingredient in batch:
            if ingredient in recipe_text:
                num_found += 1
            if num_found >= majority:
                return True, recipe_text
    return False, None


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
        num_recipes = 5
        print("Gathering the " + str(num_recipes) + " random recipes...")
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
#        print("Generated the following recipes: ")
#        for r in recipes:
#            print(str(r))

        # Now generate the hierarchical linkage from the ingredients
#        variables, labels, sparse_matrix, matrix =\
#                        __generate_model_structures(recipes, table, testing=True)
#        linkage, labels = __generate_linkage(recipes, table, testing=True)
#
#        # Now do something interesting with the linkage
#        print("Generating dendrogram...")
#        row_dendr = dendrogram(linkage, labels=labels)
#
#        print("Plotting it...")
#        plt.tight_layout()
#        plt.ylabel("Distance")
#        plt.show()

        # Clean up
        os.remove(unique_within)
        os.remove(unique)

    debug.print_test_banner(test_name, True)


def __normalize_rows_test():
    """
    Tests the normalize rows function.
    """
    test_name = "Normalize Rows Test"
    debug.print_test_banner(test_name, False)

    matrix = [
                [0, 1, 0],
                [2, 0, 1],
                [3, 1, 4],
                [0, 1, 1],
                [0, 0, 0]
             ]
    expected = [
                [0, 1, 0],
                [0.894427, 0, 0.447214],
                [0.588348, 0.1966116, 0.784465],
                [0, 0.707107, 0.707107],
                [0, 0, 0]
               ]

    matrix = np.matrix(matrix)
    expected = np.matrix(expected)

    result = __normalize_rows(matrix)

    print("Normalized : " + os.linesep + str(matrix))
    print("Got back : " + os.linesep + str(result))
    print("Expected : " + os.linesep + str(expected))

    debug.print_test_banner(test_name, True)

def __train_kmeans(matrix, sparse_matrix):
    """
    Trains k-means model.
    @param matrix: The data matrix or None if sparse matrix to be used instead
    @param sparse_matrix: The sparse data matrix (numpy sparse format), to
                          be used in case out of memory when trying to repopulate
                          the dense one.
    @return: The trained model, which it saves
    """
    if os.path.exists(config.KMEANS_MODEL_PATH):
        print("    |-> Found an existing model for kmeans at " +\
            str(config.KMEANS_MODEL_PATH) + ", using that.")
        kmeans = myio.load_pickle(config.KMEANS_MODEL_PATH)
    else:
        print("    |-> Generating the kmeans model...")
        print("    |-> Started at " + myio.print_time())
        kmeans = KMeans(n_clusters=6, verbose=1)
        if matrix is not None:
            kmeans.fit(matrix)
        else:
            kmeans.fit(sparse_matrix)
        print("    |-> Ended at " + myio.print_time())

        print("    |-> Saving the model...")
        myio.save_pickle(kmeans, config.KMEANS_MODEL_PATH)

    return kmeans


def __train_word2vec(ingredient_table, ingredients_lists):
    """
    Trains word2vec on the recipe file found in config.py.
    @param ingredient_table: The ingredient table to use for training.
    @param ingredients_lists: The list of list of ingredients.
    @return: the trained model
    """
    if os.path.exists(config.WORD2VEC_MODEL_PATH):
        print("    |-> Found an existing model for word2vec at " +\
                str(config.WORD2VEC_MODEL_PATH) + ", using that.")
        model = gensim.models.Word2Vec.load(config.WORD2VEC_MODEL_PATH)
    else:
        print("    |-> Generating the sentence generator...")
        sentences = SentenceIterator(config.RECIPE_FILE_SINGLE_PATH)

        print("    |-> Generating the ingredients generator...")
        generated_ingredient_lists = IngredientsGenerator(ingredients_lists)

        print("    |-> Training Word2Vec on ingredient file to learn " +\
                        "grammatical associations...")
        print("    |-> Started at " + myio.print_time())
        model = gensim.models.Word2Vec(sentences, min_count=1, workers=4,
                                        iter=5)
        print("    |-> Ended at " + myio.print_time())

        print("    |-> Training Word2Vec on ingredient lists to learn " +\
                        "food associations...")
        print("    |-> Started at " + myio.print_time())
        model.train(generated_ingredient_lists,
                        total_examples=len(generated_ingredient_lists))
        print("    |-> Ended at " + myio.print_time())

        print("    |-> Saving model as " + str(config.WORD2VEC_MODEL_PATH))
        model.save(config.WORD2VEC_MODEL_PATH)

    return model





class IngredientsGenerator:
    def __init__(self, ingredients_lists):
        self.recipes = ingredients_lists

    def __iter__(self):
        print("        |-> Iterating over ingredients lists...")
        for recipe in self.recipes:
            debug.debug_print("YIELDING: " + str(recipe))
            yield recipe

    def __len__(self):
        length = 0
        for recipe in self.recipes:
            length += len(recipe)
        return length



class SentenceIterator:
    def __init__(self, file_path):
        self.path = file_path
        print("        |-> Reading the recipe file into memory...")
        with open(self.path) as file_text:
            lines = [line for line in file_text]
            self.sentences = [line.split() for line in lines]
            s = []
            for sentence in self.sentences:
                sentence = [word.lower().strip(string.punctuation.replace("_", ""))\
                                for word in sentence]
                sentence = [word for word in sentence\
                                if word != config.NEW_RECIPE_LINE.lower()]
                s.append(sentence)
            self.sentences = list(s)

    def __iter__(self):
        print("        |-> Iterating over text file...")
        for sentence in self.sentences:
            #debug.debug_print("YIELDING: " + str(sentence))
            yield sentence





