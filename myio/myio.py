"""
Module to provide a place to gather my random io functions.
"""

import time
import mmap
import re
import os
import pickle
import chef_global.debug as debug

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


def copy_file(src, dest):
    """
    Overwrites dest with src.
    """
    s = open(src, 'r')
    d = open(dest, 'w')
    for line in s:
        d.write(line)
    s.close()
    d.close()


def count_occurrences(path, item):
    """
    Counts how many times the given item occurs in the
    given file and returns that number.
    @param path: The path to the file of interest
    @param item: A regular expression to find.
    @return: The number of times the given item occurs.
    """
    with open(path, 'r') as f:
        data = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        re_item = re.compile(item.encode("utf-8"))
        all_matches = re.findall(re_item, data)
        return len(all_matches)


def encode_file(path, encoding="utf-8"):
    """
    Prepends the line: # -*- coding: <encoding> -*-
    to the beginning of the given file.
    @param path: The path to the file of interest.
    @param encoding: The encoding to specify.
    @return: void
    """
    coding_line = "# -*- coding: " + str(encoding) + " -*-"
    with open(path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(coding_line + os.linesep + content)


def file_contains(path, item):
    """
    Opens the path and searches the file for the given
    regex. If it exists, returns True, otherwise False.
    @param path: The file path
    @param item: The regular expression to search for
    @return: True if item is in file, otherwise False
    """
    return len(count_occurrences(path, item)) > 0


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


def get_lines_between_tags(file_path, tag):
    """
    Generator that returns all lines between two instances of
    tag found in the file for each time that it finds another tag.
    Blah
    bloop
    TAG
    bloop
    TAG
    would return as a generator: [blah, bloop] then [bloop]
    Note that you probably want a tag at the end of the file, as
    it will otherwise just skip the last lines.
    @param file_path: The path to the file
    @param tag: The tag to look between
    """
    f = open(file_path, 'r')
    next_yield = []
    for line in f:
        if line.rstrip() == tag:
            yield next_yield
            next_yield = []
        else:
            next_yield.append(line.rstrip())
    f.close()


def get_number_of_lines(path):
    """
    Gets the total number of lines in the file.
    @param path: The path to the file
    @return: The number of lines in the file
    """
    num_lines = 0
    f = open(path, 'r')
    for line in f:
        num_lines += 1
    f.close()

    return num_lines


def join_all_files(list_of_file_paths, new_file_path):
    """
    Joins all the files in the list by opening each one
    one at a time, and copying its contents onto the
    end of new_file.
    @param list_of_file_paths: The list of file paths to use
    @param new_file_path: The path to the file that will be
                          appended to. If it doesn't exist,
                          it will be created.
    @return: void
    """
    for path in list_of_file_paths:
        append_to_file(path, new_file_path)


def load_pickle(path):
    """
    Loads the pickled item from the path and returns it.
    @param path: The pickled item.
    @return: The unpickled item
    """
    pfile = open(path, 'rb')
    to_ret = pickle.load(pfile)
    pfile.close()
    return to_ret


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


def print_time():
    """
    Returns the current time as a string.
    @return: void
    """
    return str(time.strftime("%I:%M:%S"))


def save_pickle(item, path):
    """
    Pickles the item into the given path.
    To retrieve, use load_pickle.
    @param item: The item to pickle
    @param path: The path to pickle the item into
    @return: void
    """
    output = open(path, 'wb')
    pickle.dump(item, output)
    output.close()


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


def write_list_to_file(path, the_list):
    """
    Writes the given list to the given file by appending an
    os.linesep between each item.
    @param path: The file path
    @param the_list: The iterable to write to the file
    @return: void
    """
    f = open(path, 'w')
    for item in the_list:
        f.write(str(item) + os.linesep)
    f.close()







