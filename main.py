"""
The main file. Start the program with 'python3 main.py'.
"""
import argparse
import statistics.statistics as statistics
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
        preprocessor.run_unit_tests()
        statistics.run_unit_tests()
        return "Done with unit tests."

    # if any of the args are a preprocessor command, use the preprocessor:
    if args.trim or args.reset or args.prep or args.tabulate:
        preprocessor.execute_commands(args)
        return "Preprocessor ran successfully."

    if args.train:
        statistics.train_models(args)
        return "Models have been trained"

    if args.similar:
        statistics.ask_similar(args)
        return "Similarity calculation complete."


def get_valid_args():
    """
    Validates passed-in args and returns them as something...
    @return: args TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--tabulate", help="Tabulate the ingredients from the " +
                             "list of recipes - helpful to trim it first.",
                             metavar="DATA_DIR")
    parser.add_argument("-m", "--train", help="Train the machine learning models.",
                              nargs=3, metavar=("INGREDIENT_TABLE", "UNIQUE", "UNIQUE_WITHIN"))
    parser.add_argument("-p", "--prep", help="Preprocessor pipeline the data - " +
                             "run the data through the whole preprocessor pipeline.",
                             nargs=2, metavar="DATA_DIR")
    parser.add_argument("-r", "--reset", help="Reset the data directory " +
                             "based on the given master copy directory.",
                              nargs=2, metavar=("MASTER_DIR", "DATA_DIR"))
    parser.add_argument("-t", "--trim", help="Trim the recipe data down " +
                             "to just recipes and ingredients. You must " +
                             "also specify the data directory.", metavar="DATA_DIR")
    parser.add_argument("-s", "--similar", help="Get n ingredients that are similar to " +
                              "the given list of ingredients.", nargs='+', metavar=("N", "LIST"))
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


def main():
    """
    Run the main function.
    """
    args = get_valid_args()
    result = execute_based_on_args(args)
    print_result(result)

if __name__ == "__main__":
    main()






