from sklearn.metrics import confusion_matrix
import model.models as MODELS
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, cross_val_predict
from model.result_writer import ResultWriter
from model.scoring_metrics import SCORING
import numpy as np
import pickle, os


MODELS = MODELS.MODELS

def performance_test(X, y, filename_without_extension,
                     simple, applied, seed_size=100):

    random_seeds = np.random.randint(1, 987654321, size=seed_size)
    writer = ResultWriter(filename_without_extension=filename_without_extension,
                          X=X, y=y)

    for model in MODELS:
        print(f'Training {model["name"]}')

        ## find best estimator
        grid = GridSearchCV(
            estimator=model['estimator'],
            param_grid= {} if simple else model['hyperparameters'],
            scoring=SCORING,
            cv= 5 ,
            refit='f1'  # among multiple scores, set f1 as criterion
        )
        grid.fit(X, y)
        estimator = grid.best_estimator_
        writer.save_results(estimator=estimator, model = model)

        ## init results dictionary
        result = {k: 0.0 for k in SCORING.keys()}
        result['confusion_matrix'] = np.zeros((2, 2))

        ## iterating random_seeds
        for random_seed in tqdm(random_seeds):
            cv= StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

            ## for numetric metrics
            cv_results = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, return_estimator=True)
            for k in SCORING.keys():
                result[k] += cv_results[f'test_{k}'].mean()
            ## for confusion matrix
            y_pred = cross_val_predict(estimator, X, y, cv=cv)
            result['confusion_matrix'] += confusion_matrix(y_true= y, y_pred=y_pred)

        ## save averaged results
        for k in result.keys():
            result[k] /= seed_size
        result['confusion_matrix']= np.round(result['confusion_matrix']).astype(int)

        ## write_results
        result['model_name'] = model['name']
        result['best_params'] = grid.best_params_
        writer.write_results(result=result)

        ## save fitted model
        if model['name']=="GradientBoostingClassifier":
            results_file = os.path.join("trained_model",
                                        f"model_{'applied' if applied else 'approved'}.pkl")
            with open(results_file, 'wb') as file:
                pickle.dump(grid, file)

