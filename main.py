"""
The main file. Start the program with 'python3 main.py'.
"""
import argparse
import preprocessing.preprocessing as preprocessor
import chef_global.config
from chef_global.debug import debug_print

def execute_based_on_args(args):
    """
    Executes the commands of the user based on what args they
    passed in.
    @param args: The arguments the user passed in
    @return: a printable result
    """
    # Check if debug messages should be printed as we go
    if args.verbose:
        chef_global.config.VERBOSE = True

    if args.unit_test:
        # Just do the unit tests and quit
        debug_print("Running unit tests...")
        # TODO: find all the python files in the project and run them
        # TODO: look up how to do this
        return "testing..."

    # if any of the args are a preprocessor command, use the preprocessor:
    if args.trim or args.reset:
        preprocessor.execute_commands(args)

def get_valid_args():
    """
    Validates passed-in args and returns them as something...
    @return: args TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reset", help="Reset the data directory " +
                             "based on the given master copy directory.",
                              nargs=2, metavar=("MASTER_DIR", "DATA_DIR"))
    parser.add_argument("-t", "--trim", help="Trim the recipe data down " +
                             "to just recipes and ingredients. You must " +
                             "also specify the data directory.", metavar="DATA_DIR")
    parser.add_argument("-u", "--unit_test", help="Run the unit tests " +
                             "for each module.", action="store_true")
    parser.add_argument("-v", "--verbose", help="Set verbose debug output.",
                            action="store_true")
    return parser.parse_args()

def print_result(result):
    """
    Prints the result of the program.
    @param result: The result of program execution.
    @return: void
    """
    if result is not None:
        print(result)
    else:
        print("The program didn't produce any displayable results.")

if __name__ == "__main__":
    args = get_valid_args()
    result = execute_based_on_args(args)
    print_result(result)






