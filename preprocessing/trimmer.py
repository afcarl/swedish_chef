"""
This preprocessing module provides all methods associated with
trimming and parsing the text files of the data.
"""

from tqdm import tqdm
import os
import re
import string
import shutil
from chef_global.adt import BoolTable
import myio.myio as myio
import preprocessing.prep_global as prep
import chef_global.debug as debug
import chef_global.config as config

# Ingredient file
__ing_tmp = config.DATA_DIRECTORY + "/" + "ing_tmp"

# Unique ingredient file
ingredient_file_name = config.DATA_DIRECTORY + "/" + "unique.txt"

# Unique within recipe ingredient file
within_file_name = config.DATA_DIRECTORY + "/" + "unique_within.txt"



def _clean_ingredient_test():
    """
    Run the __clean_ingredient_file method with some fake data and print the results.
    @return: void
    """
    test_data = ["<tag color=blue>blah de bloop</tag>", "most delICIOUS ingredient!",
                 "VERY GOOD PIE", "really good stuff", "wieners (the best you can get)",
                 "(cookies)", "peanut butter, bathed in clams.", "money...", "...",
                 "pie...1/2", "beep"]
    clean_data = ["blah de bloop", "most delicious ingredient", "very good pie",
                  "really good stuff", "wieners (the best you can get)", "(cookies)",
                   "peanut butter, bathed in clams", "money", "", "", "beep"]
    __unit_test(test_data, clean_data, "clean_ingredient", __clean_ingredient_file)


def _get_recipe_at_index(index, recipe_file_path):
    """
    Gets all the lines between new recipe tag at index - 1
    to new recipe tag at index from the recipe_file given.
    @param index: The index of the new recipe tag.
    @param recipe_file_path: The recipe file to look through.
    @return: The lines.
    """
    debug.debug_print("Recipe file: " + str(recipe_file_path))
    recipe_file = open(recipe_file_path, 'r')
    line_location = 0

    # First, read the file until we find the new recipe tag at index - 1
    if index == 0:
        # if index is 0, just use the start of the file as index - 1
        pass
    else:
        count_of_new_file_line = 0
        # Read the file until we find the right new recipe line
        debug.debug_print("Looking for new recipe index " + str(index - 1) + "...")
        while count_of_new_file_line != index - 1:
            for line in recipe_file:
                line_location += 1
                if line.strip() == config.NEW_RECIPE_LINE.lower():
                    count_of_new_file_line += 1
                    if count_of_new_file_line == index - 1:
                        debug.debug_print("Found right index at file line " + str(line_location))
                        break
    recipe_file.close()

    debug.debug_print("Now retrieving lines...")
    number_of_lines = myio.get_number_of_lines(recipe_file_path)
    recipe_file = open(recipe_file_path, 'r')
    debug.debug_print("Spooling to line " + str(line_location) + " out of a total " + str(number_of_lines))
    recipe = []
    found = False
    for line_number, line in enumerate(recipe_file):
        if found:
            debug.debug_print("Gathering ingredient: " + str(line.strip()) + " at line " + str(line_number))
            if line.strip() == config.NEW_RECIPE_LINE.lower():
                recipe_file.close()
                debug.debug_print("Found the recipe. Returning: ")
                debug.debug_print(str(recipe))
                return recipe
            else:
                recipe.append(line.strip())
        elif line_number == line_location:
            found = True
            debug.debug_print("Found line " + str(line_location))

    # If we have gotten here, something went wrong
    raise ValueError("Logic error. There's a bug in this method.")


def _prepare_tabulate_ingredients():
    """
    Parses the ingredients out of each file (which may or may not be trimmed, but
    trimmed files are more likely to produce results, and it will be faster) and produces
    two files: a unique.txt and a unique_within.txt, which are the unique ingredients
    and the recipe ingredients with duplicates within each recipe removed.
    After this method, the unique.txt can be used to tabulate all the ingredients and do
    basic statistics on it.
    @return: void
    """
    print("Parsing ingredients...")
    prep._apply_func_to_each_data_file(__parse_ingredients, print_info=True)

    print("Cleaning ingredients...")
    __clean_ingredient_file()

    print("Replacing 'yelk' with 'yolk'...")
    myio.find_replace(__ing_tmp, "yelk", "yolk")

    print("Removing duplicate ingredients...")
    __remove_duplicates_between_bounds(__ing_tmp, config.NEW_RECIPE_LINE.lower(),
                                        [config.NEW_RECIPE_LINE.lower()])

    print("Collapsing obvious plurals...")
    __remove_plurals(__ing_tmp)

    print("Removing empty recipes...")
    __remove_empty_recipes(__ing_tmp)

    print("Removing blank lines...")
    myio.strip_file(__ing_tmp)

    # Now make a unique_within.txt that is all the recipes with duplicate
    # ingredients removed and a unique.txt, which is all the unique ingredients
    print("Creating a unique_within.txt and a unique.txt...")
    print("    |-> Renaming " + str(__ing_tmp) + " to " + str(within_file_name))
    os.rename(__ing_tmp, within_file_name)

    print("    |-> Copying " + str(within_file_name) + " to " + \
            str(ingredient_file_name))
    shutil.copy(within_file_name, ingredient_file_name)

    print("    |-> Removing duplicates within recipes in " + \
            str(ingredient_file_name))
    __remove_duplicates_between_bounds(ingredient_file_name, "SUPER_FAKE_BOUND", [])

    print("    |-> Removing the new recipe lines from " + str(ingredient_file_name))
    myio.find_replace(ingredient_file_name, config.NEW_RECIPE_LINE.lower(), "")

    print("    |-> Removing blank lines from " + str(ingredient_file_name))
    myio.strip_file(ingredient_file_name)

    # Now tell the config file where you put the ingredient files
    print("    |-> Updating the config file to reflect location for " + \
            str(ingredient_file_name) + " and " + str(within_file_name))
    config.UNIQUE = ingredient_file_name
    config.UNIQUE_WITHIN = within_file_name


def _remove_duplicates_between_bounds_test():
    """
    Unit test for __remove_duplicates_between_bounds.
    """
    test_data =   ["cream", "pie", "cream pie", "cream", "butter", "TEST_BOUND", "cream",
                   "cream", "pie", "butter", "cheese", "TEST_BOUND", "TEST_BOUND",
                   "money", "dollars", "children", "children", "children", "dollars"]
    answer_data = ["cream", "pie", "cream pie", "",      "butter", "TEST_BOUND", "cream",
                   "",      "pie", "butter", "cheese", "TEST_BOUND", "TEST_BOUND",
                   "money", "dollars", "children", "", "", "", ""]

    __unit_test(test_data, answer_data, "remove_duplicates",
                __remove_duplicates_between_bounds, "TEST_BOUND", ["TEST_BOUND"])


def _remove_plurals_test():
    """
    Test for __remove_plurals method.
    @return: void
    """
    test_data = ["money", "moneys", "moneys", "money", "fun", "funds", "funs"]
    answer_data = ["money", "", "", "money", "fun", "funds", ""]
    __unit_test(test_data, answer_data, "remove_plurals", __remove_plurals)


def _remove_xml_from_file(path):
    """
    Removes all xml tags from the given file at path.
    @param path: The path to the file to clean.
    @return: void
    """
    text = re.sub('<[^<]+>', "", open(path).read())
    with open(path, "w") as f:
        f.write(text)


def _replace_all_ingredients_with_single_words(recipe_file_path, unique_path):
    """
    For each ingredient in unique,
    replace all instances of it in the recipe_file with
    it as a single word with underscores.
    E.g.: Feathered goose -> Feathered_goose
    @param recipe_file_path: the path to the recipe file
    @param unique_path: the path to the ingredients
    @return: void
    """
    s = ""
    with open(recipe_file_path) as recipe_file:
        s = recipe_file.read()

    ingredients = []
    with open(unique_path) as unique:
        ingredients = [line.strip().lower() for line in unique]

    for ingredient in tqdm(ingredients):
        ingredient_is_more_than_one_word = " " in ingredient
        if ingredient_is_more_than_one_word:
            ingredient_as_one_word = ingredient.replace(" ", "_")
            s = s.replace(ingredient, ingredient_as_one_word)

    with open(config.RECIPE_FILE_SINGLE_PATH, 'w') as recipe_file:
       recipe_file.write(s)


def _trim_all_files_to_recipes():
    """
    Trims away all the non-recipe, non-ingredient stuff from the data files.
    @return: void
    """
    prep._apply_func_to_each_data_file(__trim_non_recipe, print_info=True)






def __clean_ingredient_file(f=None):
    """
    Cleans up the ing_tmp file so that it no longer contains ingredients with xml tags,
    uppercase letters, or trailing punctuation.
    @param f: Optional file to read from (otherwise just uses __ing_tmp).
    @return: void
    """
    ing_file = open(__ing_tmp, 'r') if not f else open(f, 'r')
    tmp_tmp = open("tmp_tmp", 'w')

    has_numbers = lambda x: any(char.isdigit() for char in x)

    for dirty_ingredient in tqdm(ing_file):
        debug.debug_print("Cleaning " + str(dirty_ingredient) + "...")
        clean_ingredient = dirty_ingredient
        clean_ingredient = __remove_xml(clean_ingredient)
        clean_ingredient = "" if has_numbers(clean_ingredient) else clean_ingredient
        clean_ingredient = clean_ingredient.lower()
        clean_ingredient = clean_ingredient.rstrip()
        clean_ingredient = clean_ingredient.strip(
                                    string.punctuation.replace(")", "").replace("(", ""))
        clean_ingredient = clean_ingredient.lstrip(string.punctuation.replace("(", ""))
        clean_ingredient = clean_ingredient.rstrip(string.punctuation.replace(")", ""))
        clean_ingredient = clean_ingredient + os.linesep
        debug.debug_print("Writing clean ingredient " + str(clean_ingredient.rstrip()) + "...")
        tmp_tmp.write(clean_ingredient)
    ing_file.close()
    tmp_tmp.close()

    tmp_tmp = open("tmp_tmp", 'r')
    ing_file = open(__ing_tmp, 'w') if not f else open(f, 'w')

    for clean_ingredient in tmp_tmp:
        ing_file.write(clean_ingredient)
    ing_file.close()
    tmp_tmp.close()
    os.remove("tmp_tmp")


def __parse_between_tags(file_path, start_tag, stop_tag,
                         tmp_file_path, append=False, append_tag="", keep=""):
    """
    Parses out a chunk of text from the given file between the start tag and the
    stop tag and puts that text into the given tmp_file.
    @param file_path: The path to the file to parse
    @param start_tag: The start tag
    @param stop_tag: The stop tag
    @param tmp_file_path: The path to the tmp file to write to
    @param append: Whether or not to append to the tmp file (if not, overwrite)
    @param append_tag: An optional string to add to the tmp_file after each parsed item.
    @param keep: An optional string to keep instances of from original file
    @return: void
    """
    append_or_overwrite = 'a' if append else 'w'
    cookbook_file = open(file_path, 'r')
    tmp_file = open(tmp_file_path, append_or_overwrite)
    recipe_lines = []
    record = False

    buf = ""
    tmp_buf = ""
    recording = False
    # New algorithm
    # Take in the input file char by char, read into a buffer until that buffer
    # is longer than 100 chars or the start tag is in the buffer. Either way, purge it.
    # If the start tag was in the buffer though, we need to start saving the chars as we
    # read them, until we reach the end tag, at which point we need to strip the buffer
    # of the end tag and write the resulting buffer to the tmp file. Then continue.
    for line_from_original in cookbook_file:
        for char in line_from_original:
            if recording:
                tmp_buf += char
                if stop_tag in tmp_buf:
                    tmp_buf = tmp_buf[:-len(stop_tag)]
                    debug.debug_print("Found stop tag. Writing to file: " + tmp_buf)
                    to_write = tmp_buf + os.linesep + append_tag + os.linesep
                    to_write = to_write.strip() + os.linesep
                    tmp_file.write(to_write)
                    tmp_buf = ""
                    recording = False
            else:
                buf += char
                if buf[-len(start_tag):] == start_tag:
                    debug.debug_print("Found start tag...")
                    recording = True
                    buf = ""
                elif buf[-len(keep):] == keep:
                    debug.debug_print("Found a keeper")
                    tmp_file.write(keep + os.linesep)
                elif len(buf) > 10000:
                    debug.debug_print("Purging buffer")
                    buf = ""
    cookbook_file.close()
    tmp_file.close()


def __parse_ingredients(cookbook_file_path):
    """
    Takes a cookbook data file path and parses it for its ingredients, storing
    them as a list in ing_tmp by appending them to the end of it.
    @param cookbook_file_path: A path to a cookbook data file (encoded in XML),
                               may or may not be trimmed.
    @return: void
    """
    debug.debug_print("Attempting to parse " + str(cookbook_file_path) +
                             " for ingredients.")
    __parse_between_tags(cookbook_file_path, "<ingredient>", "</ingredient>",
                         __ing_tmp, append=True, keep=config.NEW_RECIPE_LINE)


def __remove_duplicates_between_bounds(file_path, bound, exceptions):
    """
    Searches the file at file_path for bounds, treating the start of the file
    as one, and removes duplicate lines within those bounds. So:

    cream                               cream
    money                               money
    cream
    cream
    peanuts                             peanuts
    BOUND                               BOUND
    cream      would turn into -->      cream
    money                               money
    spinach                             spinach
    waffles                             waffles
    cream
    BOUND                               BOUND

    @param file_path: The path to the file that will be searched
    @param bound: The bound
    @param exceptions: All the exceptions to keep regardless of repeats
    @return: void
    """
    all_lines_to_keep = []
    lines_to_keep = []
    f = open(file_path, 'r')
    for line in tqdm(f):
        if line.rstrip() == bound.rstrip():
            debug.debug_print("Found a bound, adding lines...")
            lines_to_keep.append(line)
            all_lines_to_keep.extend(lines_to_keep)
            lines_to_keep = []
        elif line in lines_to_keep and line not in exceptions:
            debug.debug_print("Found a repeat, skipping it. Repeat was: " + line.rstrip())
            lines_to_keep.append(os.linesep)
        else:
            debug.debug_print("Keeping line: " + line.rstrip())
            lines_to_keep.append(line)

    all_lines_to_keep.extend(lines_to_keep)
    f.close()
    f = open(file_path, 'w')
    for line in all_lines_to_keep:
        f.write(line)
    f.close()


def __remove_empty_recipes(file_path):
    """
    Removes empty recipes from the file at file_path.
    @param file_path: The path to the ingredient file.
    @return: void
    """
    f = open(file_path, 'r')
    # Read each line into a buffer,
    # if you come across a new_recipe line,
    # and if the line before it was new_recipe_line,
    # ignore the line

    f_tmp = open(file_path + "TMP_REMOVE_EMPTY", 'w')

    last_line = ""
    for line in f:
        if line.strip() == last_line and last_line == config.NEW_RECIPE_LINE.lower():
            pass
        else:
            last_line = line.strip()
            f_tmp.write(line)
    f.close()
    f_tmp.close()

    myio.copy_file(file_path + "TMP_REMOVE_EMPTY", file_path)
    os.remove(file_path + "TMP_REMOVE_EMPTY")


def __remove_plurals(file_path):
    """
    Removes obvious pluarls (those things that end in s, but which have counterparts
    which don't).
    @param file_path: The path to the ingredient file.
    @return: void
    """
    # This method assumes that the file is small enough to read into memory
    f = open(file_path, 'r')
    print("    |-> reading ingredients into memory...")
    ingredients = [line.rstrip() for line in f]
    f.close()

    print("    |-> searching for plurals...")
    keep = []
    already_removed = BoolTable()
    for ingredient in tqdm(ingredients):
        item_no_s = None
        already_exists_no_s = False
        if ingredient.endswith("s"):
            # Check if 1) we have already found and removed this before
            # or       2) we have an ingredient that is exactly this but without the final s
            item_no_s = ingredient[:-1]
            debug.debug_print("        |-> checking for both '" + ingredient +
                                "' and '" + item_no_s + "'...")
            if already_removed.has(ingredient):
                debug.debug_print("        |-> already removed '" + ingredient +
                                    "', do it again.")
                already_exists_no_s = True
            elif item_no_s in ingredients:
                debug.debug_print("        |-> found '" + item_no_s + "', so remove '" +
                                    ingredient + "'.")
                already_exists_no_s = True
                already_removed.put(ingredient)
        if already_exists_no_s:
            debug.debug_print("        |-> --Removing " + ingredient + "--")
            keep.append("")
        else:
            keep.append(ingredient)

    print("    |-> Writing remaining ingredients to file...")
    f = open(file_path, 'w')
    for ingredient in keep:
        f.write(ingredient + os.linesep)
    f.close()


def __remove_xml(s):
    """
    Removes xml tags from the given string.
    @param s: The string to remove xml from.
    @return: (str) s with xml stuff removed.
    """
    s = re.sub("<[^>]*>", "", s)

    return s


def __trim_non_recipe(cookbook_file_path):
    """
    Takes a cookbook data file path and removes all the non-recipe, non-ingredient
    info from it.
    @param cookbook_file_path: A path to a cookbook data file (encoded in XML)
    @return: void
    """
    debug.debug_print("Attempting to trim " + str(cookbook_file_path))
    tmp_path = "TEMPORARY_FILE_____"
    __parse_between_tags(cookbook_file_path, "<recipe", "</recipe>",
                                tmp_path, append=False, append_tag=config.NEW_RECIPE_LINE)
    myio.overwrite_file_contents(tmp_path, cookbook_file_path)
    os.remove(tmp_path)


def __unit_test(test_data, answer_data, test_name, test_function, *args):
    """
    Test framework for trimmer functions.
    @param test_data: the data to test test_function on by writing to
                      a test file and having the function read it.
    @param answer_data: The data that should come out after the test
    @param test_name: the name of the test
    @param test_function: the function to test. It MUST take a file as its
                          first parameter.
    @param args: any other arguments to the test_function after the file
                 name
    @return: void
    """
    debug.print_test_banner(test_name, False)
    dummy_file_path = test_name + ".test"

    myio.write_list_to_file(dummy_file_path, test_data)

    if len(args) > 0:
        l = [dummy_file_path]
        l.extend(args)
        func_args = tuple(l)
        test_function(*func_args)
    else:
        test_function(dummy_file_path)

    dummy_file = open(dummy_file_path, 'r')
    for i, line in enumerate(dummy_file):
        result = (line.rstrip() == answer_data[i])
        res = "passed" if result else "FAILED"
        print("Test " + str(i) + ": " + res + " " + "expected: " + answer_data[i] +
                    " --> got: " + line.rstrip())
    dummy_file.close()
    os.remove(dummy_file_path)

    debug.print_test_banner(test_name, True)


