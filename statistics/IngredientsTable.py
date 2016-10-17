"""
Module to hold the class IngredientsTable.
"""

import os
import pickle
import myio.myio as myio
import chef_global.debug as debug

class IngredientsTable:
    """
    Class to hold the master list of unique ingredients.
    Provides methods for accessing this data and for serializing it
    (that is, saving and restoring it to/from disk).
    """

    def __init__(self, ingredient_file_path):
        """
        Constructor.
        This class creates a dictionary of str->int : ingredient->uniqueID.
        @param ingredient_file_path: The path to the file that will be
                                     this object's ingredients or a saved version
                                     of this class.
        """
        self.__table = {}
        self.__next_int = 0
        self.__construct_from_file(ingredient_file_path)


    def get_id(self, ingredient):
        """
        If the ingredient is in the table, this method returns
        the unique id associated with it.
        If the ingredient is not in the table, it raises a KeyError.
        @param ingredient: The ingredient whose ID you want
        @return: The unique ID
        @throws: KeyError when ingredient not present in table
        """
        return self.__table[ingredient]


    def has(self, ingredient):
        """
        Returns whether or not the ingredient is already in the table.
        @param ingredient: The ingredient to check for
        @return: True if ingredient is already in table, False otherwise
        """
        return ingredient in self.__table


    def put(self, ingredient):
        """
        If the ingredient is not already in the table, this method updates
        the table to include it and assigns it a unique ID, which it returns.
        If the ingredient is already in the table, it returns None.
        @param ingredient: The ingredient to put in the table
        @return: None if already in table, unique id otherwise
        """
        if self.has(ingredient):
            return None
        else:
            self.__table[ingredient] = self.__next_int
            to_ret = self.__next_int
            self.__next_int += 1
            return to_ret

    def save_to_disk(self, path="ingredient_table"):
        """
        Saves this object to the disk in the given path - it overwrites
        the given file if it exists.
        @param path: The path to save at
        @return: The path it was saved to
        """
        pickle.dump(self, open(path, 'wb'))
        return path

    def __construct_from_file(self, path):
        """
        Constructs the table from the given ingredient file.
        @param path: The path to the ingredient file
        @return: void
        """
        # First try loading from a pickle
        try:
            self = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(str(e)) # TODO: catch whatever exception this prints out rather than catchall

            debug.debug_print("Constructing IngredientsTable from " + path + "...")
            f = open(path, 'r')
            for line in f:
                ingredient = line.rstrip()
                debug.debug_print("Adding '" + ingredient + "' to table.")
                self.put(ingredient)
            f.close()



def unit_test(path=None):
    """
    Runs the unit test for this module.
    @param path: The path to the test file - need a file that has some
                 dummy data in it to test with. If none is provided,
                 this function will make one itself.
    """
    debug.print_test_banner("IngredientsTable Test", False)

    dummy_path = "__INGREDIENTS_TABLE_DUMMY_DATA__"
    path = dummy_path if not path else path
    dummy_data = (path == dummy_path)

    try:
        if dummy_data:
            # Set up the dummy data first
            test_data = ["cookies", "milk", "money", "rum", "more money", "chocolate"]
            myio.write_list_to_file(path, test_data)

        uut = IngredientsTable(path)

        has_cookies = uut.has("cookies")
        has_milk = uut.has("milk")
        has_friendship = uut.has("friendship")

        cookies_id = uut.get_id("cookies")
        milk_id = uut.get_id("milk")
        chocolate_id = uut.get_id("chocolate")
        try:
            friendship_id = uut.get_id("friendship")
        except KeyError:
            friendship_id = None

        print("Pickling the unit under test...")
        pickle_path = uut.save_to_disk()
        print("Saved to: " + str(pickle_path))

        print("Loading the unit under test back from pickle...")
        uut = IngredientsTable(pickle_path)
        print("Checking if successful...")

        enemy_id = uut.put("enemy")


        # Collect and print results
        test_print = "TEST "
        if has_cookies:
            print(test_print + "0: passed -> has cookies")
        else:
            print(test_print + "0: FAILED -> no cookies")

        if has_milk:
            print(test_print + "1: passed -> has milk")
        else:
            print(test_print + "1: FAILED -> no milk")

        if not has_friendship:
            print(test_print + "2: passed -> no friendship")
        else:
            print(test_print + "2: FAILED -> has friendship")

        if cookies_id == 0:
            print(test_print + "3: passed -> cookies ID is 0")
        else:
            print(test_print + "3: FAILED -> cookies ID is " + str(cookies_id))

        if milk_id == 1:
            print(test_print + "4: passed -> milk ID is 1")
        else:
            print(test_print + "4: FAILED -> milk ID is " + str(milk_id))

        if chocolate_id == 5:
            print(test_print + "5: passed -> chocolate ID is 5")
        else:
            print(test_print + "5: FAILED -> chocolate ID is " + str(chocolate_id))

        if friendship_id is None:
            print(test_print + "6: passed -> friendship ID is None")
        else:
            print(test_print + "6: FAILED -> friendship ID is " + str(friendship_id))

        if enemy_id == 6:
            print(test_print + "7: passed -> enemy ID is 7")
        else:
            print(test_print + "7: FAILED -> enemy ID is " + str(enemy_id))
    except Exception as e:
        print(str(e))
        raise e
    finally:
        debug.print_test_banner("IngredientsTable Test", True)
        if dummy_data:
            os.remove(path)












