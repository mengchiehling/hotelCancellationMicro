import importlib
from typing import Union, List
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


def cross_validation(algorithm: str, data: pd.DataFrame, x_labels: List[str], y_label: str, optimization: bool,
                     test_size: float, configuration: str, **kwargs) -> Union[float, np.ndarray]:

    """
    Cross validation training process

    Args:
        data: training data
        x_labels: dataframe columns for features
        y_label: target vairable for supervised machine learning
        optimization: if it is a hyperparameter optimization process
        test_size: percentage of data size for algorithm early stopping
        configuration: features configuration
      len(eval_dataset)  kwargs: additional hyperparameters for the algorithm
    """

    module = importlib.import_module(f'train.logic.classification.training_process_{algorithm}')
    process = getattr(module, 'process')

    X = data
    y = data[y_label]

    y_pred = []
    y_true = []

    # ToDo: Switch to time series split
    kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    for train_index, test_index in kfold.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train = X_train[y_label]
        y_test = X_test[y_label]

        model = process(X_train[x_labels], y_train, test_size=test_size, configuration=configuration, **kwargs)

        y_temp = model.predict(X_test[x_labels])

        y_true.extend(y_test.tolist())
        y_pred.extend(y_temp.tolist())

    acc = accuracy_score(np.array(y_true), np.array(y_pred))

    if optimization:
        return acc
    else:
        return np.array(y_pred)