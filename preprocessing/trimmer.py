"""
This preprocessing module provides all methods associated with
trimming and parsing the text files of the data.
"""

import os
import myio.myio as myio
import preprocessing.prep_global as prep
import chef_global.debug as debug

def __parse_between_tags(file_path, start_tag, stop_tag, tmp_file_path, append=False):
    """
    Parses out a chunk of text from the given file between the start tag and the
    stop tag and puts that text into the given tmp_file.
    @param file_path: The path to the file to parse
    @param start_tag: The start tag
    @param stop_tag: The stop tag
    @param tmp_file_path: The path to the tmp file to write to
    @param append: Whether or not to append to the tmp file (if not, overwrite)
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
                    tmp_file.write(tmp_buf + os.linesep)
                    tmp_buf = ""
                    recording = False
            else:
                buf += char
                if buf[-len(start_tag):] == start_tag:
                    debug.debug_print("Found start tag...")
                    recording = True
                    buf = ""
                elif len(buf) > 10000:
                    debug.debug_print("Purging buffer")
                    buf = ""
    cookbook_file.close()
    tmp_file.close()
    return

    # Old algorithm
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
                if start_tag in line_from_recipe and stop_tag in line_from_recipe:
                    # If the line has BOTH the start and stop, then we only want a part of the
                    # line. TODO
                    # Keep in mind that the line may contain SEVERAL starts and stops.
                    raise NotImplementedError("This line has both a stop and a start tag on " +
                                              "it, which is not yet implemented. TODO")
                else:
                    # Otherwise, just write the line
                    debug.debug_print("Writing " + str(line_from_recipe) + "...")
                    tmp_file.write(line_from_recipe + os.linesep)
            recipe_lines = []

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
    debug.debug_print("Attempting to parse " + str(cookbook_file_path) + " for ingredients.")
    tmp_path = "ing_tmp"
    __parse_between_tags(cookbook_file_path, "<ingredient>", "</ingredient>",
                         tmp_path, append=True)
    myio.append_to_file(tmp_path, cookbook_file_path)


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
    myio.overwrite_file_contents(tmp_path, cookbook_file_path)
    os.remove(tmp_path)


def _tabulate_ingredients():
    """
    Parses the ingredients out of each file (which may or may not be trimmed, but
    trimmed files are more likely to produce results, and it will be faster). Then
    pickles the resulting dictionary and sets the config file to know about the
    dictionary.
    """
    prep._apply_func_to_each_data_file(__parse_ingredients)


def _trim_all_files_to_recipes():
    """
    Trims away all the non-recipe, non-ingredient stuff from the data files.
    @return: void
    """
    prep._apply_func_to_each_data_file(__trim_non_recipe)







