"""
A module for very simple ADTs
"""

class BoolTable:
    """
    A class to represent a dictionary with just present/not present functionality.
    Mostly useful for not having to worry about stupid keyvalue errors.
    """

    def __init__(self, *args):
        """
        Constructor.
        @param args: Optional variable list of items to put into the table.
        """
        # Internal state is really just a dictionary
        self.__table = {}

        for a in args:
            self.__table[a] = True

    def has(self, item):
        """
        Checks if the BoolTable has the given item.
        @param item: The item to check for in the table.
        @return: True if present, False if not.
        """
        try:
            return self.__table[item]
        except KeyError:
            return False

    def put(self, item):
        """
        Puts the given item into the table.
        @param item: The item to put into the table.
        @return: void
        """
        if self.has(item):
            return
        else:
            self.__table[item] = True











