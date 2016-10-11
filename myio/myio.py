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
