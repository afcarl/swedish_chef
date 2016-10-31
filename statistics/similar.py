"""
This module computes similarity between ingredients.
"""

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


def _compute_similarity_matrix(ingredients):
    """
    Computes and returns the similarity matrix
    for the given ingredients using models that
    have already been trained.
    @param ingredients: The ingredients of interest.
    @return: The similarity matrix
    """
    w2v = __load_model(config.WORD2VEC_MODEL_PATH)

    sim_mat = np.matrix([[0.0 for ing in ingredients]\
                        for ing in ingredients])

    for i, row in enumerate(sim_mat):
        row = row.A1
        for j in range(len(row)):
            similarity = w2v.similarity(ingredients[i], ingredients[j])
            sim_mat[i, j] = similarity

    df = pandas.DataFrame(sim_mat, columns=ingredients, index=ingredients)
    print(str(df))
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
    @return: The standard deviation and mean (also prints them)
    """
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
            score = _compute_similarity_score(rec)
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


def _compute_similarity_measure(ingredients):
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
    if config.SIM_STAND_DEV == 0 or config.SIM_MEAN == 0:
        print("Doesn't look like standard deviation or " +\
              "the mean of similarities make sense, you need "+\
              "to run the model generator first.")
        exit(0)
    else:
        similarity = _compute_similarity_score(ingredients)
        return (similarity_score - config.SIM_MEAN) / config.SIM_STAND_DEV


def _compute_similarity_score(ingredients):
    """
    Computes an average similarity for all the ingredients
    in the given list and returns it.
    @param ingredients: The list of ingredients.
    @return: The similarity score.
    """
    debug.debug_print("Compute similarity score...")
    w2v = __load_model(config.WORD2VEC_MODEL_PATH)
    try:
        debug.debug_print("Ingredients: " + str(ingredients))
    except KeyError:
        pass

    combos_already_seen = []
    num_scores = 0
    score = 0.0
    for ingredient in ingredients:
        debug.debug_print("Ingredient: " + str(ingredient))
        for other in ingredients:
            debug.debug_print("Other: " + str(other))
            combo = (ingredient, other)
            debug.debug_print("Combo: " + str(combo))
            already_seen = combo in combos_already_seen
            ingredient_and_other_are_same = ingredient == other
            if already_seen or ingredient_and_other_are_same:
                debug.debug_print("Already seen this combo.")
                pass
            else:
                try:
                    similarity = w2v.similarity(ingredient, other)
                    score += similarity
                    combos_already_seen.append(combo)
                    # Also append the reverse of the combo
                    combos_already_seen.append((other, ingredient))
                    num_scores += 1
                    debug.debug_print("num_scores: " + str(num_scores))
                except KeyError as e:
                    print(str(e))

    if num_scores == 0:
        return None
    else:
        return score / num_scores


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
    model = myio.load_pickle(path)
    return model









