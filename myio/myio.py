"""
Module to provide a place to gather my random io functions.
"""

import os

def append_to_file(src_path, dest_path):
    """
    Take the contents of src_path and write them into dest_path, appending
    them to the end.
    @param src_path: The path to the source file
    @param dest_path: The path to the destination file
    @return: void
    """
    src = open(src_path, 'r')
    dest = open(dest_path, 'a')
    for line in src:
        dest.write(line + os.linesep)
    src.close()
    dest.close()


def find_replace(file_path_to_search, to_replace, replace_with):
    """
    Search through file_path_to_search, find all instances of to_replace, and replace them with
    replace_with.
    @param file_path_to_search: The file path to the file to search.
    @param to_replace: The item to replace
    @param replae_with: The item to replace 'to_replace' with
    @return: void
    """
    tmp_path = "__myio_tmp__"
    file_to_search = open(file_path_to_search, 'r')
    tmp_file = open(tmp_path, 'w')
    for line in file_to_search:
        new_line = line.replace(to_replace, replace_with)
        tmp_file.write(new_line)
    file_to_search.close()
    tmp_file.close()

    file_to_search = open(file_path_to_search, 'w')
    tmp_file = open(tmp_path, 'r')
    for line in tmp_file:
        file_to_search.write(line)
    file_to_search.close()
    tmp_file.close()

    os.remove(tmp_path)


def overwrite_file_contents(src_path, dest_path):
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


def strip_file(path):
    """
    Strips the given file of all blank lines.
    @param path: The path to the file you want to strip.
    @return: void
    """
    f = open(path, 'r')
    tmp_path = "__myio__tmp__.txt"
    tmp = open(tmp_path, 'w')

    for line in f:
        if line.rstrip() != "":
            tmp.write(line)

    f.close()
    tmp.close()

    f = open(path, 'w')
    tmp = open(tmp_path, 'r')

    for line in tmp:
        f.write(line)

    f.close()
    tmp.close()
    os.remove(tmp_path)










