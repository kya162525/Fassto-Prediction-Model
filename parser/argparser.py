import argparse
import os, re

"""
    Checking command validity
        1. Check if at least one file name comes in as arguments
        2. Check if all file names all valid
    INPUT
        - parse (argparse.ArgumentParser Object)	
    RETURN 
        - None
"""
def command_process(parser):

    args = parser.parse_args()
    # Check if at least one filename is provided
    if not args.filenames:
        parser.error('At least one filename is required.')

    # Check if iteration size is Positive Integer
    if args.iteration_size <= 0:
        parser.error("Iteration size must be an positive integer.")


    # Check if all filenames are valid
    filedir = os.path.join('data', 'preprocessed') if args.preprocessed \
        else os.path.join('data', 'raw')

    for filename in args.filenames:
        if not os.path.isfile(os.path.join(filedir, filename)):
            parser.error(f'File {filename} does not exist in {filedir} directory.')

    # Print message for simple option
    if args.simple:
        print("GridSearch optimization is disabled and default hyperparameters will be used.")

    if args.applied:
        print("Applied(신청 시점) Model is used.")
    else:
        print("Approved(승인 시점) Model is used.")

    # If all checks passed, return the args object
    return args

def parse_args():
    parser = argparse.ArgumentParser(description='Testing Files')
    parser.add_argument('filenames', metavar='filenames', nargs='*',
                        help='input files to process')
    parser.add_argument('-s', '--simple', dest='simple', action='store_true',
                        help = 'user default hyperparamters without GridSearch')
    parser.add_argument('-p', '--preprocessed', dest='preprocessed', action='store_true',
                        help='use preprocessed data')
    parser.add_argument('-i', '--iteration', dest="iteration_size", type=int,
                        default = 10, help="Number of random_seed iterations")
    parser.add_argument('-b', '--applied', dest="applied", action='store_true',
                        help="applied if True else false")

    args = command_process(parser)
    return args

