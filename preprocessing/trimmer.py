"""
This preprocessing module provides all methods associated with
trimming the text files of the data.
"""

import os
import preprocessing.prep_global as prep
import chef_global.debug as debug

def __parse_between_tags(file_path, start_tag, stop_tag, tmp_file_path):
    """
    Parses out a chunk of text from the given file between the start tag and the
    stop tag and puts that text into the given tmp_file.
    @param file_path: The path to the file to parse
    @param start_tag: The start tag
    @param stop_tag: The stop tag
    @param tmp_file_path: The path to the tmp file to write to
    @return: void
    """
    cookbook_file = open(file_path, 'r')
    tmp_file = open(tmp_file_path, 'w')
    recipe_lines = []
    record = False

    for line_from_original in cookbook_file:
        if start_tag in line_from_original:
            record = True
            debug.debug_print("Recording...")

        if record:
            recipe_lines.append(line_from_original)

        if stop_tag in line_from_original:
            debug.debug_print("Hit stop tag.")
            record = False
            for line_from_recipe in recipe_lines:
                debug.debug_print("Writing " + str(line_from_recipe) + "...")
                tmp_file.write(line_from_recipe + os.linesep)
            recipe_lines = []

    cookbook_file.close()
    tmp_file.close()


def __overwrite_file_contents(src_path, dest_path):
    """
    Take the contents of src_path and write them into dest_path, after erasing
    all lines in dest_path.
    @param src_path: The path to the source file
    @param dest_path: The path to the destination file
    @return: void
    """
    src = open(src_path, 'r')
    dest = open(dest_path, 'w')
    for line in src:
        dest.write(line + os.linesep)
    src.close()
    dest.close()


def __trim_non_recipe(cookbook_file_path):
    """
    Takes a cookbook data file path and removes all the non-recipe, non-ingredient
    info from it.
    @param cookbook_file_path: A path to a cookbook data file (encoded in XML)
    @return: void
    """
    debug.debug_print("Attempting to trim " + str(cookbook_file_path))
    tmp_path = "tmp"
    __parse_between_tags(cookbook_file_path, "<recipe", "</recipe>", tmp_path)
    __overwrite_file_contents(tmp_path, cookbook_file_path)
    os.remove(tmp_path)


def _trim_all_files_to_recipes():
    """
    Trims away all the non-recipe, non-ingredient stuff from the data files.
    @return: void
    """
    prep._apply_func_to_each_data_file(__trim_non_recipe)
