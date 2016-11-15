"""
Place to put some useful cross-module utilities.
"""

import chef_global.debug as debug
import chef_global.config as config
import string


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





