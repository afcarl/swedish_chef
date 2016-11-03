"""
Module to provide a means to accelerate slow python code.
"""

import multiprocessing


def acc_map(func, items):
    """
    Applies func to each item in items.
    @param func: A function of the form f(item)
    @param items: The list of items to apply func to
    @return: void
    """
    p = multiprocessing.Pool(None)
    ret = p.map(func, items)
    return ret

def f(x):
    return x * x

if __name__ == "__main__":
    m = acc_map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(str(m))
