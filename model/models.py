from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


MODELS = [
    {
        'name': 'GradientBoostingClassifier',
        'estimator': GradientBoostingClassifier(),
        'hyperparameters': {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 8],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        },
        'attr':['feature_importances_']
    }
]
"""
{
    'name': 'LogisticRegressionClassifier',
    'estimator': LogisticRegression(),
    'hyperparameters': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000, 5000]
    },
    'attr': ['coef_']
},
{
    'name': 'RandomForestClassifier',
    'estimator': RandomForestClassifier(n_jobs=-1),
    'hyperparameters': {
        'n_estimators': [100, 500],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    'attr':['feature_importances_']
},
"""
