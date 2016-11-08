"""
This is the module responsible for generating new recipes.
"""

from tqdm import tqdm
import myio.myio as myio
import chef_global.config as config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def _encode_bit_vector(bit_vector):
    """
    Encodes the given vector into a compressed
    format and returns the compressed version.
    @param bit_vector: A vector of 1s and 0s, which is mostly 0s.
    @return: The compressed version of the bit vector.
    """
    # TODO: Apply Golomb's or IRL on the bit vector
    pass


def _encode_training_data(rec_table):
    """
    Encodes all of the ingredients into their compressed
    format and pairs them into lists that belong together
    according to recipes, then tuples that with the recipe
    itself.
    @param rec_table: A fully loaded RecipeTable object
                      that has recipes with ingredients
                      and text.
    @return:void, but a list of tuples of the form
            ([encoded ingredients], recipe) are written
            to disk.
    """
    training_data = []
    for recipe in tqdm(rec_table):
        encoded_ingredients = [_encode_bit_vector(ingredient)
                                for ingredient in recipe]
        data_point = (encoded_ingredients, recipe)
        training_data.append(data_point)
    myio.save_pickle(training_data, config.TRAINING_PATH)



def _generate_recipe(ingredients, rec_table):
    """
    The main API function for this module.
    Takes a list of ingredients and a recipe table.
    Turns the list of ingredients into feature vectors,
    then compresses them into much smaller vectors and
    feeds those into the trained RNN.
    Prints the recipe that gets generated.
    @param ingredients: A list of ingredients to use in the recipe
    @param rec_table: A RecipeTable object that can be used to convert
                      the ingredients to feature vectors.
    @return: void
    """
    feature_vectors = [rec_table.ingredient_to_feature_vector(ingredient)
                            for ingredient in ingredients]

    encoded_fvs = [_encode_bit_vector(fv) for fv in feature_vectors]

    generated_recipe = __get_recipe_from_rnn(encoded_fvs)

    print("Generated this recipe for you: ")
    print(str(generated_recipe))


def _train_rnn(rec_table):
    """
    Loads the training data from disk and uses it
    to train the RNN.
    @param rec_table: A fully loaded RecipeTable object
    @return: void
    """
    # TODO
    rnn = MyRNN()
    pass



def __get_recipe_from_rnn(encoded_feature_vectors):
    """
    Feeds the given bit vectors into the neural network
    and has it generate a recipe.
    @param encoded_feature_vectors: A list of encoded ingredients
                                    to use in the recipe.
    @return: The generated recipe
    """
    training_data = myio.load_pickle(config.TRAINING_PATH)
    # TODO
    pass



class MyRNN:
    """
    RNN class.
    """
    def __init__(self):
        pass


