import pickle
import pandas as pd
from parser.argparser import parse_args
from preproc.preprocessor import PreProcessor
import os

"""
	customer.csv like filename only without "in_yn"
"""
if __name__ == '__main__':
	args = parse_args()
	if args.preprocessed:
		file_path = os.path.join("data", "preprocessed", args.filenames[0])
		new_data = pd.read_csv(file_path, index_col='cst_cd')
	else:
		file_path = os.path.join("data", "raw")

		print("Data Loading..")
		cs = pd.read_csv(os.path.join(file_path, args.filenames[0]))
		ga = pd.read_csv(os.path.join(file_path, args.filenames[1]), low_memory=False)
		user = pd.read_csv(os.path.join(file_path, args.filenames[2]))

		print("PreProcessing..")
		processor = PreProcessor(cs_df=cs, ga_df=ga, user_df=user, predict=True)
		new_data = processor.get_preprocessed_data(args.applied, predict=True)

		file_path = os.path.join("data", "preprocessed", args.filenames[0])
		new_data.to_csv(file_path)

	results_file = os.path.join("trained_model",
				 f"model_{'applied' if args.applied else 'approved'}.pkl")
	with open(results_file, 'rb') as file:
		model = pickle.load(file)

	predictions = model.predict_proba(new_data)
	df = pd.DataFrame(
		{
			'cst_cd' : new_data.index,
			'pred': predictions[:,1]
		}
	)
	df.to_csv(os.path.join("result", f"pred_{args.filenames[0]}" ),index=False)

