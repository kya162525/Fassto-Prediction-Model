import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, precision_score, \
    log_loss, fbeta_score



def precision_score_with_zero_division(y_true, y_pred, average):
    return precision_score(y_true, y_pred, zero_division=1, average=average)

def recall_score_with_zero_division(y_true, y_pred, average):
    return recall_score(y_true, y_pred, zero_division=1, average=average)

def f1_score_with_zero_division(y_true, y_pred, average):
    return f1_score(y_true, y_pred, zero_division=1, average=average)

def cross_entropy_with_zero_division(y_true, y_pred, **kwargs):
    try:
        return log_loss(y_true, y_pred, **kwargs)
    except ZeroDivisionError:
        return np.inf

SCORING = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score_with_zero_division, average='macro'),
           'recall': make_scorer(recall_score_with_zero_division, average='macro'),
           'f1': make_scorer(f1_score_with_zero_division, average='macro'),
           'f-beta': make_scorer(fbeta_score, beta=2, average='macro'),
           'cross_entropy': make_scorer(cross_entropy_with_zero_division, needs_proba=True)
           }

