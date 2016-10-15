"""
This preprocessing module provides all methods associated with
trimming and parsing the text files of the data.
"""

import os
import re
import string
import myio.myio as myio
import preprocessing.prep_global as prep
import chef_global.debug as debug
import chef_global.config as config

# Ingredient file
__ing_tmp = "ing_tmp"

# New cookbook marker
__new_cookbook_line = "NEW_COOKBOOK______________________________LINE"

# New recipe marker
__new_recipe_line = "NEW_RECIPE__________________________________LINE"


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

    dummy_file = open(dummy_file_path, 'w')
    for d in test_data:
        dummy_file.write(d + os.linesep)
    dummy_file.close()

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


def _clean_ingredient_test():
    """
    Run the __clean_ingredient_file method with some fake data and print the results.
    @return: void
    """
    test_data = ["<tag color=blue>blah de bloop</tag>", "most delICIOUS ingredient!",
                 "VERY GOOD PIE", "really good stuff", "wieners (the best you can get)",
                 "(cookies)", "peanut butter, bathed in clams.", "money...", "..."]
    clean_data = ["blah de bloop", "most delicious ingredient", "very good pie",
                  "really good stuff", "wieners (the best you can get)", "(cookies)",
                   "peanut butter, bathed in clams", "money", ""]
    __unit_test(test_data, clean_data, "clean_ingredient", __clean_ingredient_file)


def __clean_ingredient_file(f=None):
    """
    Cleans up the ing_tmp file so that it no longer contains ingredients with xml tags,
    uppercase letters, or trailing punctuation.
    @param f: Optional file to read from (otherwise just uses __ing_tmp).
    @return: void
    """
    ing_file = open(__ing_tmp, 'r') if not f else open(f, 'r')
    tmp_tmp = open("tmp_tmp", 'w')

    for dirty_ingredient in ing_file:
        debug.debug_print("Cleaning " + str(dirty_ingredient) + "...")
        clean_ingredient = dirty_ingredient
        clean_ingredient = __remove_xml(clean_ingredient)
        clean_ingredient = clean_ingredient.lower()
        clean_ingredient = clean_ingredient.rstrip()
        clean_ingredient = clean_ingredient.strip(
                                    string.punctuation.replace(")", "").replace("(", ""))
        clean_ingredient = clean_ingredient.lstrip(string.punctuation.replace("(", ""))
        clean_ingredient = clean_ingredient.rstrip(string.punctuation.replace(")", ""))
        clean_ingredient = clean_ingredient + os.linesep
        debug.debug_print("Writing clean ingredient " + str(clean_ingredient) + "...")
        tmp_tmp.write(clean_ingredient)
    ing_file.close()
    tmp_tmp.close()

    tmp_tmp = open("tmp_tmp", 'r')
    ing_file = open(__ing_tmp, 'w') if not f else open(f, 'w')

    for clean_ingredient in tmp_tmp:
        if clean_ingredient.strip() != "":
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
                         __ing_tmp, append=True, keep=__new_recipe_line)



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
    for line in f:
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
    tmp_path = "tmp"
    __parse_between_tags(cookbook_file_path, "<recipe", "</recipe>",
                                tmp_path, append=False, append_tag=__new_recipe_line)
    myio.overwrite_file_contents(tmp_path, cookbook_file_path)
    os.remove(tmp_path)


def _tabulate_ingredients():
    """
    Parses the ingredients out of each file (which may or may not be trimmed, but
    trimmed files are more likely to produce results, and it will be faster). Then
    pickles the resulting dictionary and sets the config file to know about the
    dictionary.
    """
    print("Parsing ingredients...")
    prep._apply_func_to_each_data_file(__parse_ingredients)

    print("Cleaning ingredients...")
    __clean_ingredient_file()

    print("Replacing 'yelk' with 'yolk'...")
    myio.find_replace(__ing_tmp, "yelk", "yolk")

    print("Removing duplicate ingredients...")
    __remove_duplicates_between_bounds(__ing_tmp, __new_recipe_line.lower(),
                                        [__new_recipe_line.lower()])

    print("Removing blank lines...")
    myio.strip_file(__ing_tmp)


def _trim_all_files_to_recipes():
    """
    Trims away all the non-recipe, non-ingredient stuff from the data files.
    @return: void
    """
    prep._apply_func_to_each_data_file(__trim_non_recipe)







