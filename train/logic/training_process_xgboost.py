import os
from typing import Union

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src import config

from src.config import features_configuration


def build_transformer_pipeline(stats=None) -> Union[ColumnTransformer, Pipeline]:

    """

    Returns:
        a scikit-learn pipeline
    """

    transformers = []
    for key, value in features_configuration.items():
        if key == 'numerical':
            transformers.append((key, 'passthrough', features_configuration[key]))
        elif key == 'onehot':
            transformers.append((key, OneHotEncoder(handle_unknown="ignore"), features_configuration[key]))

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
    positive_records = y.sum()
    negative_records = len(y) - positive_records
    spw = negative_records / positive_records

    random_seed = 42
    feature_extractor = build_transformer_pipeline(stats=chi2)

    model = Pipeline([('feature_extractor', feature_extractor),
                       ('model', XGBClassifier( random_state=random_seed,
                                                n_estimators=3000, objective='binary:logistic', scale_pos_weight=spw,n_jobs=1))
                       ])

    k = kwargs.get('k', None)
    if k:
        k = int(k)
        model.set_params(**{f"feature_extractor__feature_selector__k": k})

    reg_alpha = kwargs['reg_alpha']
    reg_lambda = kwargs['reg_lambda']
    min_child_weight = kwargs['min_child_weight']
    learning_rate = kwargs['learning_rate']
    max_leaves = int(kwargs['max_leaves'])

    model.set_params(**{"model__reg_alpha": reg_alpha,
                        "model__min_child_weight": min_child_weight,
                        "model__learning_rate": learning_rate,
                        "model__max_leaves": max_leaves,
                        "model__reg_lambda": reg_lambda,
                        })

    y = y.astype(np.int32).values
    x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=test_size, random_state=random_seed)

    try:
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(x_train, y_train)
        eval_set = [(model_temp.transform(x_val), y_val)]
    except ValueError:
        model.set_params(**{f"feature_extractor__feature_selector__k": 'all'})
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(x_train, y_train)
        eval_set = [(model_temp.transform(x_val), y_val)]

    # For example, setting it to 100 means we stop the training if the predictions have not improved for
    # the last 100 rounds.
    # https://stackoverflow.com/questions/40329576/sklearn-pass-fit-parameters-to-xgboost-in-pipeline/55711752#55711752

    model.fit(x_train, y_train, model__eval_set=eval_set, model__early_stopping_rounds=100, model__eval_metric="logloss")

    return model
