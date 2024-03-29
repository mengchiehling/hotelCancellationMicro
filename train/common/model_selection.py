import importlib
from typing import Union, List
from functools import partial

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score

from src import config


def cross_validation(data: pd.DataFrame,  y_label: str, optimization: bool,
                     test_size: float, **kwargs) -> Union[float, np.ndarray]:

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

    module = importlib.import_module(f'train.logic.training_process_{config.algorithm}')
    process = getattr(module, 'process')

    x = data
    y = data[y_label]

    date_list = x['check_in'].values.tolist()
    date_list.sort()
    date_start = date_list[0]
    date_end = date_list[-1]
    idx = pd.date_range(date_start, date_end)
    idx = [t.strftime("%Y-%m-%d") for t in idx]
    date_feature = pd.DataFrame(data=np.zeros([len(idx), 1]), index=idx)

    y_pred = []
    y_true = []

    kfold = TimeSeriesSplit(n_splits=5, test_size=None, max_train_size=None)

    for n_fold, (train_index, test_index) in enumerate(kfold.split(date_feature)):

        train_time = date_feature.iloc[train_index].index
        test_time = date_feature.iloc[test_index].index

        x_train = x[x['check_in'].isin(train_time)]
        x_test = x[x['check_in'].isin(test_time)]

        y_train = x_train[y_label]
        y_test = x_test[y_label]

        train_time_begin = date_feature.iloc[train_index[0]].name
        train_time_end = date_feature.iloc[train_index[-1]].name

        test_time_begin = date_feature.iloc[test_index[0]].name
        test_time_end = date_feature.iloc[test_index[-1]].name

        print(f"fold {n_fold}: training: {train_time_begin} - {train_time_end},"
              f"testing: {test_time_begin} - {test_time_end}")

        model = process(x_train, y_train, test_size=test_size, **kwargs)
        y_temp = model.predict(x_test)

        y_true.extend(y_test.tolist())
        y_pred.extend(y_temp.tolist())

        print(accuracy_score(y_test, y_temp))

    acc = accuracy_score(np.array(y_true), np.array(y_pred))

    if optimization:
        return acc
    else:
        return np.array(y_pred)
