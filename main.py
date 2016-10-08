"""
The main file. Start the program with 'python3 main.py'.
"""

if __name__ == "__main__":
    args = get_valid_args()
    result = execute_based_on_args(args)
    print_result(result)
