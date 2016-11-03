"""
This module computes similarity between ingredients.
"""

import random
import warnings
import statistics.ingredients_table as ingredients_table
import chef_global.debug as debug
from statistics.recipe import Recipe
import statistics.recipe_table as recipe_table
import pandas
from tqdm import tqdm
import math
import numpy as np
import chef_global.config as config
import myio.myio as myio


def _compute_similarity_matrix(ingredients, w2v=None):
    """
    Computes and returns the similarity matrix
    for the given ingredients using models that
    have already been trained.
    @param ingredients: The ingredients of interest.
    @return: The similarity matrix
    """
    if w2v is None:
        w2v = __load_model(config.WORD2VEC_MODEL_PATH)

    sim_mat = np.matrix([[0.0 for ing in ingredients]\
                        for ing in ingredients])

    for i, row in enumerate(sim_mat):
        row = row.A1
        for j in range(len(row)):
            similarity = w2v.similarity(ingredients[i], ingredients[j])
            sim_mat[i, j] = similarity

    return sim_mat


def _compute_sim_stats_test():
    """
    Test for sim stats method.
    """
    recipes = []
    table = ingredients_table.load_from_disk(config.INGREDIENT_TABLE_PATH)
    recipes.append(Recipe(table, ingredients=["apple", "pineapple"]))
    recipes.append(Recipe(table, ingredients=[]))
    recipes.append(Recipe(table, ingredients=[""]))
    recipes.append(Recipe(table, ingredients=["soup", "tuna", "sandwhich"]))
    recipes.append(Recipe(table, ingredients=["fish", "salmon", "egg", "flour", "sugar"]))
    recipes.append(Recipe(table, ingredients=["ground_beef", "duck", "tuna"]))
    debug.debug_print("Recipes: " + str(recipes))
    rec_table = recipe_table.RecipeTable(recipes)

    _compute_sim_stats(rec_table)


def _compute_sim_stats(rec_table):
    """
    Computes the standard deviation and mean
    of the similarity scores for each recipe in
    the given rec_table.
    @param rec_table: The RecipeTable object.
    @return: The mean, standard deviation
    """
    if config.SIM_MEAN is not None:
        print("Mean has already been calculated, here it is: " + str(config.SIM_MEAN))
        print("Standard deviation: " + str(config.SIM_STAND_DEV))
        return config.SIM_MEAN, config.SIM_STAND_DEV
    else:
        total = 0.0
        N = 0
        scores = []
        print("    |-> Computing the mean...")
        for rec in tqdm(rec_table):
            try:
                debug.debug_print("Recipe: " + str(rec))
            except KeyError:
                pass
            if len(rec) == 0:
                debug.debug_print("Skipping this recipe, it is empty.")
                pass
            else:
                score = _compute_similarity_score(rec, w2v=w2v)
                debug.debug_print("Similarity: " + str(score))
                scores.append(score)
                if score is not None:
                    total += score
                    N += 1
        mean = total / N

        deviances = []
        print("    |-> Computing the standard deviation...")
        for score in tqdm(scores):
            if score is not None:
                deviance = score - mean
                deviances.append(deviance)
        dev_squared = [deviance * deviance for deviance in deviances]
        sum_of_squares = sum(dev_squared)
        variance = sum_of_squares / N
        std_dev = math.sqrt(variance)

        print("You should record these values in the config.py: ")
        print("Mean: " + str(mean))
        print("Standard deviation: " + str(std_dev))
        return (mean, std_dev)


def _compute_similarity_measure(ingredients, w2v=None):
    """
    Computes a similarity measure of the ingredients. The
    similarity measure is, under the hood, a z-score of how
    computed form the ingredients' similarity score and
    the similarity mean and stand_dev found in config.py.
    @param ingredients: The ingredients of interest.
    @return: A floating point number that represents how
             many standard deviations this list of ingredients
             is from the mean in terms of similarity. This
             is a good proxy for how likely these ingredients
             are to make a good recipe.
    """
    if w2v is None:
        w2v = __load_model(config.WORD2VEC_MODEL_PATH)

    if config.SIM_STAND_DEV == 0 or config.SIM_MEAN == 0:
        print("Doesn't look like standard deviation or " +\
              "the mean of similarities make sense, you need "+\
              "to run the model generator first.")
        exit(0)
    else:
        similarity = _compute_similarity_score(ingredients, w2v=w2v)
        if similarity is None:
            return None
        else:
            return (similarity - config.SIM_MEAN) / config.SIM_STAND_DEV


def _compute_similarity_score(ingredients, w2v=None):
    """
    Computes an average similarity for all the ingredients
    in the given list and returns it.
    @param ingredients: The list of ingredients.
    @return: The similarity score.
    """
    if w2v is None:
        w2v = __load_model(config.WORD2VEC_MODEL_PATH)
    try:
        debug.debug_print("Computing sim score for ingredients: " + str(ingredients))
        pass
    except KeyError:
        pass

    combos_already_seen = []
    num_scores = 0
    score = 0.0
    for ingredient in ingredients:
        for other in ingredients:
            combo = (ingredient, other)
            already_seen = combo in combos_already_seen
            ingredient_and_other_are_same = ingredient == other
            if already_seen or ingredient_and_other_are_same:
                pass
            else:
                try:
                    similarity = w2v.similarity(ingredient, other)
                    score += similarity
                    combos_already_seen.append(combo)
                    # Also append the reverse of the combo
                    combos_already_seen.append((other, ingredient))
                    num_scores += 1
                except KeyError as e:
                    #print(str(e))
                    pass

    if num_scores == 0:
        debug.debug_print("Can't compute z score.")
        return None
    else:
        debug.debug_print("Can compute z score: " + str(score / num_scores))
        return score / num_scores


def _compute_similarity_stats(ingredients, w2v=None):
    """
    Computes a similarity matrix, a similarity score, and
    a similarity measure.
    @param ingredients: The list of ingredients to deal with
    @return: A tuple: (sim_matrix, sim_score, sim_measure)
    """
    if w2v is None:
        w2v = __load_model(config.WORD2VEC_MODEL_PATH)
    similarity_matrix = _compute_similarity_matrix(ingredients, w2v=w2v)
    similarity_score = _compute_similarity_score(ingredients, w2v=w2v)
    similarity_measure = _compute_similarity_measure(ingredients, w2v=w2v)
    return (similarity_matrix, similarity_score, similarity_measure)




def _get_random_similar_ingredients(num_ingredients, rec_table, w2v=None, seed=None):
    """
    Returns num_ingredients random ingredients that are
    'similar' to one another.
    @param num_ingredients: The number of ingredients to get.
    @param rec_table: A RecipeTable object
    @param seed: A seed for choosing the same ones everytime.
    @return: The ingredients that are similar.
    """
    if num_ingredients < 1:
        raise ValueError("Number of ingredients must be more than 0. Given: " + str(num_ingredients))

    # TODO: in desperate need of refactor
    if w2v is None:
        w2v = __load_model(config.WORD2VEC_MODEL_PATH)
    kmeans = __load_model(config.KMEANS_MODEL_PATH)

    all_ingredients_in_w2v = False
    while not all_ingredients_in_w2v:
        stored = []
        cluster = None
        while cluster is None:
            seed_ingredient = rec_table.get_random_ingredient(seed)
            if num_ingredients == 1:
                return [seed_ingredient]
            stored.append(seed_ingredient)
            if len(stored) == num_ingredients:
                break
            else:
                debug.debug_print("Got random ingredient: " + str(seed_ingredient))
                feature_vector = rec_table.ingredient_to_feature_vector(seed_ingredient)
                seed_cluster_index = (kmeans.predict(np.array(feature_vector).reshape(1, -1)))[0]
                debug.debug_print("Cluster index for this feature vector: " + str(seed_cluster_index))
                cluster = rec_table.get_cluster(seed_cluster_index)

        debug.debug_print("Stored ingredients: " + str(stored))
        stored = list(set(stored))
        ingredients = []
        if stored:
            ingredients.extend(stored)
        debug.debug_print("Ingredients after adding stored: " + str(ingredients))
        converged = False
        for j in range(1000):
            debug.debug_print("Attempting to find some similar ingredients, iteration: " + str(j))
            if cluster is not None:
                while len(ingredients) != num_ingredients:
                    index = random.randint(0, len(cluster.ingredients) - 1)
                    if cluster.ingredients[index] in ingredients:
                        pass
                    else:
                        ingredients.append(cluster.ingredients[index])
            else:
                debug.debug_print("No cluster to pull from.")
            #print("On iteration " + str(j) + " found these ingredients: " + str(ingredients))
            similarity = _compute_similarity_measure(ingredients, w2v)
            #print("Similarity for these ingredients: " + str(similarity))
            if similarity is None or similarity < 0.2:
                debug.debug_print("Did not converge on iteration: " + str(j))
                converged = False
                if len(stored) == num_ingredients:
                    # we walked through with just the random seed ingredients, and they didn't
                    # work together. Empty them out.
                    ingredients = []
                else:
                    ingredients = stored
                debug.debug_print("Trying again...")
            else:
                debug.debug_print("Converged!")
                converged = True
                break
        if not converged:
            break

        debug.debug_print("Going to attempt to calculate similarity matrix now...")
        try:
            _compute_similarity_matrix(ingredients, w2v)
            all_ingredients_in_w2v = True
            debug.debug_print("And they are all in w2v, so we can move on.")
        except KeyError:
            all_ingredients_in_w2v = False
            debug.debug_print("But they were not all in w2v, so we try again.")

    if not converged:
        print("Could not converge on " + str(num_ingredients) + " similar items.")
        return None
    else:
        return ingredients


def _get_similar_ingredients_to(ingredients, num_ingredients, rec_table):
    """
    Gets num_ingredients which are similar to ingredients.
    @param ingredients: The ingredients to compare to when getting new ones
    @param num_ingredients: The number of new ingredients to get.
    @param rec_table: The recipe table
    @return: New ingredients, which are similar to the passed in ones
    """
    w2v = __load_model(config.WORD2VEC_MODEL_PATH)

    print("    |-> Finding a seed ingredient...")
    seed_sim = 0.0
    while seed_sim is None or seed_sim < 0.2:
        seed_ingredient = (_get_random_similar_ingredients(1, rec_table, w2v=w2v))[0]
        seed_sim = _compute_similarity_measure([seed_ingredient, ingredients[0]], w2v=w2v)
        print("Got: " + str(seed_ingredient))
        print("Sim measure: " + str(seed_sim))

    print("    |-> Found seed ingredient: " + str(seed_ingredient))

    sims = [i for i in ingredients]
    sims.append(seed_ingredient)

    print("    |-> Gathering ingredients...")
    for i in range(num_ingredients - 1):
        sim = 0.0
        ingredient_already_found = False
        while sim is None or sim < 0.2 or ingredient_already_found:
            next_not_in_w2v = True
            while next_not_in_w2v:
                next_ingredient = (_get_random_similar_ingredients(1, rec_table, w2v=w2v))[0]
                try:
                    _compute_similarity_stats([next_ingredient, "water"], w2v=w2v)
                    next_not_in_w2v = False
                    ingredient_already_found = next_ingredient in sims
                except KeyError:
                    next_not_in_w2v = True

            nexts = [ing for ing in sims]
            nexts.append(next_ingredient)
            sim = _compute_similarity_measure(nexts, w2v=w2v)
        print("    |-> Found an ingredient to add: " + str(next_ingredient))
        sims.append(next_ingredient)

    for i in ingredients:
        sims.remove(i)

    print("    |-> Here are the ingredients: " + str(sims))
    return sims


def _unit_test():
    """
    Do the unit tests for this module.
    """
    _compute_sim_stats_test()


def __load_model(path):
    """
    Loads the given path and returns it as a model.
    @param path: The path to the given model.
    @return: The loaded model.
    """
    print("Loading model: " + str(path))
    model = myio.load_pickle(path)
    return model









