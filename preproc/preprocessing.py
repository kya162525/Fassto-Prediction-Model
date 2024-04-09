from preproc.tools import *
from preproc.preprocessor import *


"""
    INPUT
        - file_name 
            - if preprocessed, preprocessed .csv file
            - if not, ['cs.csv', 'ga.csv', 'user.csv']-like list
        - preprocessed
    RETURN  
        - X : 
        - y : 
        - filename_without_extension
"""
def preprocessing(file_name, preprocessed, applied):

    # preprocessed data
    if preprocessed:
        # file_name
        match = re.match(r'^(.*)\.csv$', file_name[0])
        if match:
            filename_without_extension = match.group(1)
        else:
            raise ValueError(f"Invalid file extension for {file_name[0]}. Only .csv files are allowed.")
        ## preprocessing
        file_path = os.path.join("data", "preprocessed", file_name[0])

        X = pd.read_csv(file_path, index_col='cst_cd')
        y = X.pop('is_in')

    # raw data
    else:
        # Prepare input data
        print("Data Loading..")
        file_path = os.path.join("data", "raw")
        cs_df = pd.read_csv(os.path.join(file_path, file_name[0]))
        ga_df = pd.read_csv(os.path.join(file_path, file_name[1]), low_memory=False)
        user_df = pd.read_csv(os.path.join(file_path, file_name[2]))

        # Preprocess data
        print("PreProcessing..")
        preprocessor = PreProcessor(cs_df, ga_df, user_df)
        train = preprocessor.get_preprocessed_data(applied)

        file_path = os.path.join("data", "preprocessed", f"processed_{file_name[0]}")
        train.to_csv(file_path)
        match = re.match(r'^(.*)\.csv$', file_name[1])
        filename_without_extension = match.group(1)

        X = train.copy()
        y = X.pop('is_in')

        
    return X, y, filename_without_extension
