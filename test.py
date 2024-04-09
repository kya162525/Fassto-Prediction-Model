from preproc.preprocessing import preprocessing
from model.performance_test import performance_test
from parser.argparser import parse_args

def PRINT_FORMAT(file_name, width = 30):
	print('-' * width)
	print('{:^{width}}'.format(file_name, width=width))
	print('-' * width)


if __name__ == '__main__':
	"""
	Argument Parsing
		- file_names
		- preprocessed 
		- iteration_size
		- simple (hyperparameter tuning)
	"""
	args = parse_args()

	X, y, filename_without_extension = preprocessing(args.filenames, args.preprocessed, args.applied)
	performance_test(X, y, filename_without_extension,
					seed_size=args.iteration_size, simple=args.simple, applied=args.applied)
