import os
from typing import Union

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src import config

from src import config


def build_transformer_pipeline(stats=None) -> Union[ColumnTransformer, Pipeline]:

    """

    Returns:
        a scikit-learn pipeline
    """

    transformers = []
    for key, value in config.features_configuration.items():
        if key == 'numerical':
            transformers.append((key, 'passthrough', config.features_configuration[key]))
        elif key == 'onehot':
            transformers.append((key, OneHotEncoder(handle_unknown="ignore"), config.features_configuration[key]))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    feature_transformer = Pipeline([('column_transformer', column_transformer),
                                    ('feature_selector', SelectKBest(stats))])

    return feature_transformer


def process(x: pd.Series, y: pd.Series, test_size: float, **kwargs):
    """
    For binary classfication, choose objective='binary'
    :param x:
    :param y:
    :param test_size:
    :param kwargs:
    :return:
    """
    random_seed = 42
    feature_extractor = build_transformer_pipeline(stats=chi2)

    model = Pipeline([('feature_extractor', feature_extractor),
                       ('model', SVC(random_state=random_seed, class_weight=config.class_weight, probability = True
                                                       ))])

    k = kwargs.get('k', None)
    if k:
        k = int(k)
        model.set_params(**{f"feature_extractor__feature_selector__k": k})

    c = kwargs['C']

    model.set_params(**{
                        "model__C": c
                        })

    y = y.astype(np.int32).values
    x_train = x.copy()
    y_train = y.copy()

    try:
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(x_train, y_train)
        # eval_set = [(model_temp.transform(x_train), y_train)]
    except ValueError:
        model.set_params(**{f"feature_extractor__feature_selector__k": 'all'})
        # model_temp = Pipeline(model.steps[:-1])
        # model_temp.fit_transform(x_train, y_train)
        # eval_set = [(model_temp.transform(x_train), y_train)]

    # For example, setting it to 100 means we stop the training if the predictions have not improved for
    # the last 100 rounds.
    # https://stackoverflow.com/questions/40329576/sklearn-pass-fit-parameters-to-xgboost-in-pipeline/55711752#55711752

    model.fit(x_train, y_train)

    return model
