from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict


class ResultWriter:
    def __init__(
            self,
            filename_without_extension,
            X,
            y
            ):
        self.filename_without_extension = filename_without_extension
        self.X = X; self.y = y

        # making result directory
        current_time = (datetime.now() + timedelta(hours=9)).strftime('%m%d_%H%M%S')
        self.RESULT_DIR = os.path.join("result", f"{self.filename_without_extension}_{current_time}")
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        # init results
        self.init_results()

    def __del__(self):
        self.extended.to_csv(os.path.join(self.RESULT_DIR, f"{self.filename_without_extension}_extended.csv"), index_label='cst_cd')
        self.features.to_csv(os.path.join(self.RESULT_DIR, "features.csv"), index_label='name')
        with open(os.path.join(self.RESULT_DIR,"result.txt"), "w") as file:
            file.write(self.result_str.getvalue())
            self.result_str.close()

    def init_results(self):
        # 1. extended.csv
        self.extended = pd.concat([self.X, pd.DataFrame({'is_in':self.y.values}, index=pd.Index(self.X.index))], axis=1)
        # 2. features.csv
        self.features = pd.DataFrame(columns = ['name'] + self.X.columns.to_list())
        self.features.set_index('name', inplace=True)
        # 3. result.txt
        self.result_str = StringIO()
    """
    INPUT
        result
            - acuracy, precision, recall, f1, cross_entropy, cfm(np.array(2,2))
            - model (model_name)
            - best_params_ (grid.best_params_0
    """
    def write_results(self, result):
        accuracy = result['accuracy']
        precision = result['precision']
        recall = result['recall']
        f1 = result['f1']
        cross_entropy = result['cross_entropy']
        fbeta = result['f-beta']
        cfm = result['confusion_matrix']

        with open(os.path.join(self.RESULT_DIR, "result.txt"), 'a') as f:


            self.result_str.write(f"{result['model_name']}\n")
            self.result_str.write(f"Best params : {result['best_params']}\n")
            self.result_str.write(f' -- {self.filename_without_extension} model test --\n')
            self.result_str.write(f'| Accuracy : \t {accuracy:.3f}\t\t\t|\n')
            self.result_str.write(f'| CROSS-ENTROPY: {cross_entropy:.3f}\t\t\t|\n')
            self.result_str.write(f'| PRECISION: \t {precision:.3f}\t\t\t|\n')
            self.result_str.write(f'| RECALL: \t\t {recall:.3f}\t\t\t|\n')
            self.result_str.write(f'| F1: \t\t\t {f1:.3f}\t\t\t|\n')
            self.result_str.write(f'| F-beta: \t\t {fbeta:.3f}\t\t\t|\n')

            self.result_str.write(' ' + ('-' * 55) + '\n')

            # Calculate the classification report from the confusion matrix

            y_true = np.concatenate([np.zeros(cfm[0].sum()), np.ones(cfm[1].sum())])
            y_pred = np.concatenate([np.zeros(cfm[0][0]), np.ones(cfm[0][1]),
                                     np.zeros(cfm[1][0]), np.ones(cfm[1][1])])

            for line in classification_report(y_true, y_pred, zero_division=1).split('\n'):
                self.result_str.write(f'| {line}{" " * (55 - len(line))}|' + '\n')

            self.result_str.write(' ' + ('-' * 55) + '\n')
            self.result_str.write('Confusion matrix: \n')
            self.result_str.write(f'{cfm}\n\n')

    """
    INPUT
        estimator - fitted model
        attr - ["coef_", "feature_importances_"]
        model : object in model.models.py

    """
    def save_results(self, estimator, model):

        # features.csv
        for attr in model['attr']:
            result = getattr(estimator, attr)
            while np.ndim(result) > 1:
                result = np.squeeze(result)

            new_row = pd.DataFrame({k:v for k,v in zip(self.X.columns, result)},
                                    index = [f"{model['name']}.{attr}"])
            self.features = pd.concat([self.features, new_row])

        # proba to extended
        proba_pred = cross_val_predict(estimator, self.X, self.y, cv=5, method='predict_proba')
        new_col = pd.DataFrame({f"{model['name']}_proba" : proba_pred[:,1]}, index=self.extended.index)
        self.extended = pd.concat([self.extended, new_col], axis=1)
